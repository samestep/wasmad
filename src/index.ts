import binaryen from "binaryen";

/** export name and internal name for the forward pass  */
const fwd = "fwd";

/** export name and internal name for the backward pass  */
const bwd = "bwd";

class Autodiff {
  mod: binaryen.Module;
  grad: number[];
  fwd: binaryen.ExpressionRef[];
  bwd: binaryen.ExpressionRef[];

  constructor(mod: binaryen.Module, grad: number[]) {
    this.mod = mod;
    this.grad = grad;
    this.fwd = [];
    this.bwd = [];
  }

  local(expr: binaryen.ExpressionRef): number {
    const info = binaryen.getExpressionInfo(expr);
    if (info.id !== binaryen.LocalGetId)
      throw Error("Only local.get is supported");
    return (info as binaryen.LocalGetInfo).index;
  }

  binary(info: binaryen.BinaryInfo, grad: number): binaryen.ExpressionRef {
    switch (info.op) {
      case binaryen.SubFloat64:
        const x = this.local(info.left);
        const y = this.local(info.right);
        const dx = this.grad[x];
        const dy = this.grad[y];
        this.bwd.push(
          this.mod.local.set(
            dx,
            this.mod.f64.add(
              this.mod.local.get(dx, binaryen.f64),
              this.mod.local.get(grad, binaryen.f64),
            ),
          ),
          this.mod.local.set(
            dy,
            this.mod.f64.sub(
              this.mod.local.get(dy, binaryen.f64),
              this.mod.local.get(grad, binaryen.f64),
            ),
          ),
        );
        return this.mod.f64.sub(
          this.mod.local.get(x, binaryen.f64),
          this.mod.local.get(y, binaryen.f64),
        );
      default:
        throw Error("Unsupported binary operation");
    }
  }

  expression(
    info: binaryen.ExpressionInfo,
    grad: number,
  ): binaryen.ExpressionRef {
    switch (info.id) {
      case binaryen.BinaryId:
        return this.binary(info as binaryen.BinaryInfo, grad);
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
  );
  const body = ad.expression(
    binaryen.getExpressionInfo(f.body),
    bwdParams.length,
  );

  const fwdOut = fwdParams.length;
  ad.fwd.push(
    mod.local.set(fwdOut, body),
    mod.tuple.make([
      ...results.map((_, i) =>
        mod.tuple.extract(mod.local.get(fwdOut, f.results), i),
      ),
      ...resultsGrad.map(() => mod.f64.const(0)),
      mod.i32.const(0),
    ]),
  );
  const fwdResult = binaryen.createType([
    ...results,
    ...resultsGrad,
    binaryen.i32,
  ]);
  mod.addFunction(
    fwd,
    binaryen.createType(fwdParams),
    fwdResult,
    [f.results],
    mod.block(null, ad.fwd, fwdResult),
  );
  mod.addFunctionExport(fwd, fwd);

  const bwdOut = bwdParams.length;
  ad.bwd.push(
    mod.local.set(
      bwdOut,
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
    [binaryen.createType(resultsGrad)],
    mod.block(null, ad.bwd, bwdResult),
  );
  mod.addFunctionExport(bwd, bwd);
};
