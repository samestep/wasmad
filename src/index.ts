import binaryen from "binaryen";
import { Tape, makeTapes } from "./tape.js";
import * as util from "./util.js";

const gradType = (type: binaryen.Type): binaryen.Type => {
  if (type !== binaryen.f64) throw Error("Only f64 values are supported");
  return binaryen.f64;
};

interface Var {
  /** type of the original variable */
  type: binaryen.Type;

  /** variable index in forward pass */
  fwd: number;

  /** variable index of current gradient value in backward pass */
  grad: number;
}

interface Result {
  /** expression in the forward pass */
  fwd: binaryen.ExpressionRef;

  /** local index for gradient in the backward pass */
  bwd: number;
}

class Autodiff {
  params: binaryen.Type[];

  paramsGrad: binaryen.Type[];

  results: binaryen.Type[];

  resultsGrad: binaryen.Type[];

  /** info about variables from the original function, including parameters */
  vars: Var[];

  fwdParams: binaryen.Type[];

  bwdParams: binaryen.Type[];

  /** types of variables in the forward pass, including parameters */
  fwdVars: binaryen.Type[];

  /** types of variables in the backward pass, including parameters */
  bwdVars: binaryen.Type[];

  /** forward pass expressions to save as locals, in order by field index */
  fwdFields: Map<binaryen.ExpressionRef, number>;

  bwdFields: number[];

  voidGrad: number;

  /** backward pass body, in reverse order */
  bwd: binaryen.ExpressionRef[];

  constructor(
    private mod: binaryen.Module,
    f: binaryen.FunctionInfo,
    private tape: Tape,
  ) {
    this.params = binaryen.expandType(f.params);
    this.results = binaryen.expandType(f.results);
    this.paramsGrad = this.params.map(gradType);
    this.resultsGrad = this.results.map(gradType);

    const t = util.typeFromHeapType(tape.struct, false);
    this.fwdParams = [...this.params, ...this.paramsGrad];
    this.bwdParams = [...this.paramsGrad, ...this.resultsGrad, t];

    this.fwdVars = [...this.fwdParams];
    this.bwdVars = [...this.bwdParams];

    this.vars = [
      ...this.params.map((type, index) => ({ type, fwd: index, grad: index })),
      ...f.vars.map((type) => {
        const fwd = this.fwdVars.length;
        this.fwdVars.push(type);
        const grad = this.bwdVars.length;
        this.bwdVars.push(gradType(type));
        return { type, fwd, grad };
      }),
    ];

    this.fwdFields = new Map(
      tape.fwd.map((ref) => {
        const index = this.fwdVars.length;
        this.fwdVars.push(binaryen.getExpressionType(ref));
        return [ref, index];
      }),
    );
    this.bwdFields = tape.fwd.map((ref) => {
      const index = this.bwdVars.length;
      this.bwdVars.push(binaryen.getExpressionType(ref));
      return index;
    });

    this.voidGrad = this.bwdVars.length;
    this.bwdVars.push(binaryen.createType([]));

    this.bwd = [];
  }

  makeFwd(type: binaryen.Type): number {
    const index = this.fwdVars.length;
    this.fwdVars.push(type);
    return index;
  }

  makeBwd(type: binaryen.Type): number {
    const index = this.bwdVars.length;
    this.bwdVars.push(type);
    return index;
  }

  fwdGet(index: binaryen.ExpressionRef): binaryen.ExpressionRef {
    return this.mod.local.get(index, this.fwdVars[index]);
  }

  set(expr: binaryen.ExpressionRef): {
    index: number;
    expr: binaryen.ExpressionRef;
  } {
    const index = this.makeBwd(binaryen.getExpressionType(expr));
    return { index, expr: this.mod.local.set(index, expr) };
  }

  get(index: binaryen.ExpressionRef): binaryen.ExpressionRef {
    return this.mod.local.get(index, this.bwdVars[index]);
  }

  untape(ref: binaryen.ExpressionRef): binaryen.ExpressionRef {
    return this.bwdFields[util.unwrap(this.tape.bwd.get(ref))];
  }

  block(ref: binaryen.ExpressionRef, info: binaryen.BlockInfo): Result {
    if (info.children.length === 0)
      return { fwd: this.mod.block(info.name, []), bwd: this.voidGrad };
    const last = util.unwrap(info.children.pop());
    const children = info.children.map((child) => this.expr(child).fwd);
    const { fwd, bwd } = this.expr(last);
    children.push(fwd);
    return { fwd: this.mod.block(info.name, children, info.type), bwd };
  }

  localGet(ref: binaryen.ExpressionRef, info: binaryen.LocalGetInfo): Result {
    const { fwd, grad } = this.vars[info.index];
    return { fwd: this.fwdGet(fwd), bwd: grad };
  }

  localSet(ref: binaryen.ExpressionRef, info: binaryen.LocalSetInfo): Result {
    const { fwd, bwd } = this.expr(info.value);
    const local = this.vars[info.index];
    const grad = this.makeBwd(this.bwdVars[local.grad]);
    this.bwd.push(this.mod.local.set(bwd, this.get(grad)));
    local.grad = grad;
    return info.isTee
      ? {
          fwd: this.mod.local.tee(local.fwd, fwd, this.fwdVars[local.fwd]),
          bwd: grad,
        }
      : { fwd: this.mod.local.set(local.fwd, fwd), bwd: this.voidGrad };
  }

  const(ref: binaryen.ExpressionRef, info: binaryen.ConstInfo): Result {
    if (typeof info.value !== "number")
      throw Error("Unsupported constant value");
    switch (info.type) {
      case binaryen.f64:
        return {
          fwd: this.mod.f64.const(info.value),
          bwd: this.makeBwd(binaryen.f64),
        };
      default:
        throw Error("Unsupported constant type");
    }
  }

  binary(ref: binaryen.ExpressionRef, info: binaryen.BinaryInfo): Result {
    const left = this.expr(info.left);
    const right = this.expr(info.right);
    const dx = left.bwd;
    const dy = right.bwd;
    const dz = this.makeBwd(binaryen.f64);
    switch (info.op) {
      case binaryen.AddFloat64: {
        this.bwd.push(
          this.mod.local.set(dx, this.mod.f64.add(this.get(dx), this.get(dz))),
          this.mod.local.set(dy, this.mod.f64.add(this.get(dy), this.get(dz))),
        );
        return { fwd: this.mod.f64.add(left.fwd, right.fwd), bwd: dz };
      }
      case binaryen.SubFloat64: {
        this.bwd.push(
          this.mod.local.set(dx, this.mod.f64.add(this.get(dx), this.get(dz))),
          this.mod.local.set(dy, this.mod.f64.sub(this.get(dy), this.get(dz))),
        );
        return { fwd: this.mod.f64.sub(left.fwd, right.fwd), bwd: dz };
      }
      case binaryen.MulFloat64: {
        const x = this.untape(info.left);
        const y = this.untape(info.right);
        this.bwd.push(
          this.mod.local.set(
            dx,
            this.mod.f64.add(
              this.get(dx),
              this.mod.f64.mul(this.get(dz), this.get(y)),
            ),
          ),
          this.mod.local.set(
            dy,
            this.mod.f64.add(
              this.get(dy),
              this.mod.f64.mul(this.get(dz), this.get(x)),
            ),
          ),
        );
        return { fwd: this.mod.f64.mul(left.fwd, right.fwd), bwd: dz };
      }
      case binaryen.DivFloat64: {
        const y = this.untape(info.right);
        const z = this.untape(ref);
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
        return { fwd: this.mod.f64.div(left.fwd, right.fwd), bwd: dz };
      }
      default:
        throw Error("Unsupported binary operation");
    }
  }

  expression(
    ref: binaryen.ExpressionRef,
    info: binaryen.ExpressionInfo,
  ): Result {
    switch (info.id) {
      case binaryen.BlockId:
        return this.block(ref, info as binaryen.BlockInfo);
      case binaryen.LocalGetId:
        return this.localGet(ref, info as binaryen.LocalGetInfo);
      case binaryen.LocalSetId:
        return this.localSet(ref, info as binaryen.LocalSetInfo);
      case binaryen.ConstId:
        return this.const(ref, info as binaryen.ConstInfo);
      case binaryen.BinaryId:
        return this.binary(ref, info as binaryen.BinaryInfo);
      default:
        throw Error("Unsupported expression");
    }
  }

  expr(ref: binaryen.ExpressionRef): Result {
    let { fwd, bwd } = this.expression(ref, binaryen.getExpressionInfo(ref));
    const index = this.fwdFields.get(ref);
    if (index !== undefined)
      fwd = this.mod.local.tee(index, fwd, this.fwdVars[index]);
    return { fwd, bwd };
  }
}

interface Names {
  fwd: string;
  bwd: string;
}

const makeNames = (names: string[]): Names[] => {
  const set = new Set(names);
  return names.map((name) => {
    // TODO: be smarter to avoid pathological cases, maybe e.g. like this?
    // https://cs.stackexchange.com/a/39700
    let fwd = `${name}_fwd`;
    for (let i = 1; set.has(fwd); ++i) fwd = `${name}_fwd${i}`;
    set.add(fwd);
    let bwd = `${name}_bwd`;
    for (let i = 1; set.has(bwd); ++i) bwd = `${name}_bwd${i}`;
    set.add(bwd);
    return { fwd, bwd };
  });
};

export interface Gradient {
  fwd: binaryen.FunctionRef;
  bwd: binaryen.FunctionRef;
}

/**
 * Return the forward and backward passes added to `mod` for each preexisting
 * function.
 */
export const autodiff = (mod: binaryen.Module): Gradient[] => {
  const tapes = makeTapes(mod);
  const infos = tapes.map((_, i) =>
    binaryen.getFunctionInfo(mod.getFunctionByIndex(i)),
  );
  const names = makeNames(infos.map(({ name }) => name));
  return infos.map((f, i) => {
    const tape = tapes[i];
    const { fwd: fwdName, bwd: bwdName } = names[i];

    const ad = new Autodiff(mod, f, tape);
    const tapeVar = ad.bwdParams.length - 1;
    const { fwd: fwdBody, bwd: gradResults } = ad.expr(f.body);

    const out = ad.makeFwd(f.results);
    const fwdResult = binaryen.createType([
      ...ad.results,
      ...ad.resultsGrad,
      ad.bwdVars[tapeVar],
    ]);
    const fwd = mod.addFunction(
      fwdName,
      binaryen.createType(ad.fwdParams),
      fwdResult,
      ad.fwdVars.slice(ad.fwdParams.length),
      mod.block(
        null,
        [
          mod.local.set(out, fwdBody),
          mod.tuple.make([
            ...ad.results.map((_, i) => mod.tuple.extract(ad.fwdGet(out), i)),
            ...ad.resultsGrad.map(() => mod.f64.const(0)),
            util.structNew(
              mod,
              [...ad.fwdFields.values()].map((index) => ad.fwdGet(index)),
              tape.struct,
            ),
          ]),
        ],
        fwdResult,
      ),
    );

    const bwdResult = binaryen.createType(ad.paramsGrad);
    const bwd = mod.addFunction(
      bwdName,
      binaryen.createType(ad.bwdParams),
      bwdResult,
      ad.bwdVars.slice(ad.bwdParams.length),
      mod.block(
        null,
        [
          ...ad.bwdFields.map((index, i) =>
            mod.local.set(
              index,
              util.structGet(mod, i, ad.get(tapeVar), ad.bwdVars[i]),
            ),
          ),
          mod.local.set(
            gradResults,
            mod.tuple.make(
              ad.resultsGrad.map((_, i) => ad.get(ad.paramsGrad.length + i)),
            ),
          ),
          ...ad.bwd.reverse(),
          mod.tuple.make(ad.paramsGrad.map((_, index) => ad.get(index))),
        ],
        bwdResult,
      ),
    );

    return { fwd, bwd };
  });
};
