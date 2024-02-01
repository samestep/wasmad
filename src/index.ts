import binaryen from "binaryen";

/** export name and internal name for the forward pass  */
const fwd = "fwd";

/** export name and internal name for the backward pass  */
const bwd = "bwd";

class Autodiff {
  mod: binaryen.Module;
  grad: number[];
  vars: binaryen.Type[];
  bwd: binaryen.ExpressionRef[];

  constructor(mod: binaryen.Module, grad: number[], vars: binaryen.Type[]) {
    this.mod = mod;
    this.grad = grad;
    this.vars = vars;
    this.bwd = [];
  }

  make(type: binaryen.Type): number {
    const index = this.vars.length;
    this.vars.push(type);
    return index;
  }

  set(expr: binaryen.ExpressionRef): {
    index: number;
    expr: binaryen.ExpressionRef;
  } {
    const index = this.make(binaryen.getExpressionType(expr));
    return { index, expr: this.mod.local.set(index, expr) };
  }

  get(index: binaryen.ExpressionRef): binaryen.ExpressionRef {
    return this.mod.local.get(index, this.vars[index]);
  }

  local(expr: binaryen.ExpressionRef): number {
    const info = binaryen.getExpressionInfo(expr);
    if (info.id !== binaryen.LocalGetId)
      throw Error("Only local.get is supported");
    return (info as binaryen.LocalGetInfo).index;
  }

  binary(
    info: binaryen.BinaryInfo,
    z: number,
    dz: number,
  ): binaryen.ExpressionRef {
    const x = this.local(info.left);
    const y = this.local(info.right);
    const dx = this.grad[x];
    const dy = this.grad[y];
    switch (info.op) {
      case binaryen.SubFloat64: {
        this.bwd.push(
          this.mod.local.set(dx, this.mod.f64.add(this.get(dx), this.get(dz))),
          this.mod.local.set(dy, this.mod.f64.sub(this.get(dy), this.get(dz))),
        );
        return this.mod.f64.sub(this.get(x), this.get(y));
      }
      case binaryen.DivFloat64: {
        // this code appears to set `dy` first, using `dx1` before defining it,
        // but `this.bwd` will eventually get reversed so it's fine
        const dx1 = this.set(this.mod.f64.div(this.get(dz), this.get(y)));
        this.bwd.push(
          this.mod.local.set(
            dx,
            this.mod.f64.add(this.get(dx), this.get(dx1.index)),
          ),
          this.mod.local.set(
            dy,
            this.mod.f64.sub(
              this.get(dy),
              this.mod.f64.mul(this.get(dx1.index), this.get(z)),
            ),
          ),
          dx1.expr,
        );
        return this.mod.f64.div(this.get(x), this.get(y));
      }
      default:
        throw Error("Unsupported binary operation");
    }
  }

  expression(
    info: binaryen.ExpressionInfo,
    y: number,
    dy: number,
  ): binaryen.ExpressionRef {
    switch (info.id) {
      case binaryen.BinaryId:
        return this.binary(info as binaryen.BinaryInfo, y, dy);
      default:
        throw Error("Unsupported expression");
    }
  }
}

export const autodiff = (mod: binaryen.Module) => {
  if (mod.getNumFunctions() !== 1)
    throw Error("Module must contain exactly one function");
  if (mod.hasMemory()) throw Error("Module must not contain a memory");

  const f = binaryen.getFunctionInfo(mod.getFunctionByIndex(0));
  if (f.vars.length > 0)
    throw Error("Function must not contain local variables");

  const params = binaryen.expandType(f.params);
  const results = binaryen.expandType(f.results);
  const paramsGrad = params.map((param) => {
    if (param !== binaryen.f64)
      throw Error("Only f64 parameters are supported");
    return binaryen.f64;
  });
  const resultsGrad = results.map((result) => {
    if (result !== binaryen.f64) throw Error("Only f64 results are supported");
    return binaryen.f64;
  });

  const fwdParams = [...params, ...paramsGrad];
  const bwdParams = [
    ...params,
    ...paramsGrad,
    ...results,
    ...resultsGrad,
    binaryen.i32,
  ];

  const ad = new Autodiff(
    mod,
    params.map((_, i) => params.length + i),
    [...bwdParams],
  );
  const out = ad.make(f.results);
  const grad = ad.make(binaryen.createType(resultsGrad));
  const body = ad.expression(binaryen.getExpressionInfo(f.body), out, grad);

  const fwdResult = binaryen.createType([
    ...results,
    ...resultsGrad,
    binaryen.i32,
  ]);
  mod.addFunction(
    fwd,
    binaryen.createType(fwdParams),
    fwdResult,
    ad.vars.slice(fwdParams.length),
    mod.block(
      null,
      [
        mod.local.set(out, body),
        mod.tuple.make([
          ...results.map((_, i) =>
            mod.tuple.extract(mod.local.get(out, f.results), i),
          ),
          ...resultsGrad.map(() => mod.f64.const(0)),
          mod.i32.const(0),
        ]),
      ],
      fwdResult,
    ),
  );
  mod.addFunctionExport(fwd, fwd);

  ad.bwd.push(
    mod.local.set(
      out,
      mod.tuple.make(
        results.map((_, i) =>
          mod.local.get(params.length + paramsGrad.length + i, binaryen.f64),
        ),
      ),
    ),
    mod.local.set(
      grad,
      mod.tuple.make(
        resultsGrad.map((_, i) =>
          mod.local.get(
            params.length + paramsGrad.length + results.length + i,
            binaryen.f64,
          ),
        ),
      ),
    ),
  );
  ad.bwd.reverse();
  ad.bwd.push(
    mod.tuple.make(
      paramsGrad.map((_, i) => mod.local.get(params.length + i, binaryen.f64)),
    ),
  );
  const bwdResult = binaryen.createType(paramsGrad);
  mod.addFunction(
    bwd,
    binaryen.createType(bwdParams),
    bwdResult,
    ad.vars.slice(bwdParams.length),
    mod.block(null, ad.bwd, bwdResult),
  );
  mod.addFunctionExport(bwd, bwd);
};
