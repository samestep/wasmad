import binaryen from "binaryen";
import { LoadKind, Tape, makeTapes } from "./tape.js";
import { Types, becomesMutable, unit } from "./type.js";
import * as util from "./util.js";

interface Names {
  fwd: string;
  bwd: string;
}

const makeNames = (names: string[]): Map<string, Names> => {
  const set = new util.Names();
  for (const name of names) set.add(name);
  const gradNames = new Map<string, Names>();
  for (const name of names) {
    const fwd = set.make(`${name}_fwd`);
    const bwd = set.make(`${name}_bwd`);
    gradNames.set(name, { fwd, bwd });
  }
  return gradNames;
};

interface Var {
  /** type of the original variable */
  type: binaryen.Type;

  /** primal variable index in forward pass */
  fwd: number;

  /** gradient variable index in forward pass */
  grad: number;

  /** variable index of current gradient value in backward pass */
  bwd: number;
}

interface Result {
  /** expression for primal value in the forward pass */
  fwd: binaryen.ExpressionRef;

  /** local index for gradient in the forward pass */
  grad: number;

  /** local index for gradient in the backward pass */
  bwd: number;
}

class Autodiff {
  /** types of the original function parameters */
  params: binaryen.Type[];

  /** gradient types for the original function parameters */
  paramsGrad: binaryen.Type[];

  /** types of the original function results */
  results: binaryen.Type[];

  /** gradient types for the original function results */
  resultsGrad: binaryen.Type[];

  /** info about variables from the original function, including parameters */
  vars: Var[];

  /** types of the forward pass parameters */
  fwdParams: binaryen.Type[];

  /** types of the backward pass parameters */
  bwdParams: binaryen.Type[];

  /** types of variables in the forward pass, including parameters */
  fwdVars: binaryen.Type[];

  /** types of variables in the backward pass, including parameters */
  bwdVars: binaryen.Type[];

  /** forward pass local indices for all field indices in tape struct */
  fwdFields: number[];

  /** backward pass local indices for all field indices in tape struct */
  bwdFields: number[];

  /** forward pass local index for a zero-valued variable of type `f64` */
  fwdZeroF64: number;

  /** forward pass local index for an empty tuple variable */
  fwdVoid: number;

  /** backward pass local index for an empty tuple variable */
  bwdVoid: number;

  /** backward pass body, in reverse order */
  bwd: binaryen.ExpressionRef[];

  constructor(
    private types: Types,
    private mod: binaryen.Module,
    private names: Map<string, Names>,
    f: binaryen.FunctionInfo,
    private tape: Tape,
  ) {
    this.params = binaryen.expandType(f.params);
    this.results = binaryen.expandType(f.results);
    this.paramsGrad = types.tuple(this.params);
    this.resultsGrad = types.tuple(this.results);

    const t = util.typeFromHeapType(tape.struct, false);
    this.fwdParams = [...this.params, ...this.paramsGrad];
    this.bwdParams = [...this.paramsGrad, ...this.resultsGrad, t];

    this.fwdVars = [...this.fwdParams];
    this.bwdVars = [...this.bwdParams];

    let gradIndex = 0;
    this.vars = [
      ...this.params.map((type, index) => {
        const bwd = gradIndex;
        if (types.type(type) !== unit) ++gradIndex;
        return { type, fwd: index, grad: this.params.length + bwd, bwd };
      }),
      ...f.vars.map((type) => {
        const fwd = this.fwdVars.length;
        this.fwdVars.push(type);
        const gradType = types.type(type);
        const grad = this.fwdVars.length;
        this.fwdVars.push(gradType);
        const bwd = this.bwdVars.length;
        this.bwdVars.push(gradType);
        return { type, fwd, grad, bwd };
      }),
    ];

    this.fwdFields = [];
    this.bwdFields = [];
    for (const { type } of util.structTypeGetFields(tape.struct)) {
      this.fwdFields.push(this.fwdVars.length);
      this.bwdFields.push(this.bwdVars.length);
      this.fwdVars.push(type);
      this.bwdVars.push(type);
    }

    this.fwdZeroF64 = this.fwdVars.length;
    this.fwdVars.push(binaryen.f64);

    this.fwdVoid = this.fwdVars.length;
    this.fwdVars.push(unit);

    this.bwdVoid = this.bwdVars.length;
    this.bwdVars.push(unit);

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

  get(index: binaryen.ExpressionRef): binaryen.ExpressionRef {
    return this.mod.local.get(index, this.bwdVars[index]);
  }

  /** Return backward-pass expression to get the primal value for `ref`. */
  untape(ref: binaryen.ExpressionRef): binaryen.ExpressionRef {
    const load = util.unwrap(this.tape.loads.get(ref));
    switch (load.kind) {
      case LoadKind.Const:
        return this.mod.f64.const(load.value); // TODO: other types
      case LoadKind.Field:
        return this.get(this.bwdFields[load.index]);
    }
  }

  /** Return backward-pass local index for the gradient of `ref`. */
  bwdGrad(ref: binaryen.ExpressionRef): number {
    const gradType = this.types.type(binaryen.getExpressionType(ref));
    const load = this.tape.gradLoads.get(ref);
    if (load === undefined) return this.makeBwd(gradType);
    switch (load.kind) {
      case LoadKind.Const: {
        if (load.value !== 0) throw Error("Non-zero gradient constant");
        return this.makeBwd(gradType);
      }
      case LoadKind.Field:
        return this.bwdFields[load.index];
    }
  }

  block(ref: binaryen.ExpressionRef, info: binaryen.BlockInfo): Result {
    if (info.children.length === 0)
      return {
        fwd: this.mod.block(info.name, []),
        grad: this.fwdVoid,
        bwd: this.bwdVoid,
      };
    const last = util.unwrap(info.children.pop());
    const children = info.children.map((child) => this.expr(child).fwd);
    const { fwd, grad, bwd } = this.expr(last);
    children.push(fwd);
    return { fwd: this.mod.block(info.name, children, info.type), grad, bwd };
  }

  call(ref: binaryen.ExpressionRef, info: binaryen.CallInfo): Result {
    const operands = info.operands.map((operand) => this.expr(operand));

    const { fwd: fwdName, bwd: bwdName } = util.unwrap(
      this.names.get(info.target),
    );

    const params = info.operands.map(binaryen.getExpressionType);
    const results = binaryen.expandType(info.type);
    const paramsGrad = this.types.tuple(params);
    const resultsGrad = this.types.tuple(results);
    const field = util.unwrap(this.tape.calls.get(ref));

    const fwdTape = this.fwdFields[field];
    const fwdGrad = this.makeFwd(binaryen.createType(resultsGrad));
    const returnType = binaryen.createType([
      ...results,
      ...resultsGrad,
      this.fwdVars[fwdTape],
    ]);
    const tuple = this.makeFwd(returnType);
    const call = this.mod.local.tee(
      tuple,
      this.mod.call(
        fwdName,
        [
          ...operands.map(({ fwd }) => fwd),
          ...operands.map(() => this.mod.f64.const(0)), // TODO: other types
        ],
        returnType,
      ),
      returnType,
    );

    const bwdTape = this.bwdFields[field];
    const gradIn = this.makeBwd(binaryen.createType(paramsGrad));
    const gradOut = this.makeBwd(binaryen.createType(resultsGrad));
    operands.forEach(({ bwd }, i) => {
      this.bwd.push(
        this.mod.local.set(bwd, this.mod.tuple.extract(this.get(gradIn), i)),
      );
    });
    this.bwd.push(
      this.mod.local.set(
        gradIn,
        this.mod.call(
          bwdName,
          [
            ...operands.map(({ bwd }) => this.get(bwd)),
            ...resultsGrad.map((_, i) =>
              this.mod.tuple.extract(this.get(gradOut), i),
            ),
            this.get(bwdTape),
          ],
          binaryen.createType(paramsGrad),
        ),
      ),
    );

    return {
      fwd: this.mod.block(
        null,
        [
          this.mod.local.set(
            fwdTape,
            this.mod.tuple.extract(call, results.length + resultsGrad.length),
          ),
          // TODO: handle the empty tuple case
          this.mod.local.set(
            fwdGrad,
            util.tupleMake(
              this.mod,
              resultsGrad.map((_, i) =>
                this.mod.tuple.extract(this.fwdGet(tuple), results.length + i),
              ),
            ),
          ),
          util.tupleMake(
            this.mod,
            results.map((_, i) =>
              this.mod.tuple.extract(this.fwdGet(tuple), i),
            ),
          ),
        ],
        info.type,
      ),
      grad: fwdGrad,
      bwd: gradOut,
    };
  }

  localGet(ref: binaryen.ExpressionRef, info: binaryen.LocalGetInfo): Result {
    const { fwd, grad, bwd } = this.vars[info.index];
    return { fwd: this.fwdGet(fwd), grad, bwd };
  }

  localSet(ref: binaryen.ExpressionRef, info: binaryen.LocalSetInfo): Result {
    const value = this.expr(info.value);
    const local = this.vars[info.index];
    const gradType = this.bwdVars[local.bwd];
    const bwd = this.makeBwd(gradType);
    local.bwd = bwd;
    if (gradType === unit) {
      const fwd = info.isTee
        ? this.mod.local.tee(local.fwd, value.fwd, this.fwdVars[local.fwd])
        : this.mod.local.set(local.fwd, value.fwd);
      return { fwd, grad: local.grad, bwd };
    } else {
      this.bwd.push(this.mod.local.set(value.bwd, this.get(bwd)));
      const children = [
        this.mod.local.set(local.fwd, value.fwd),
        this.mod.local.set(local.grad, this.fwdGet(value.grad)),
      ];
      if (info.isTee) children.push(this.fwdGet(local.fwd));
      return {
        fwd: this.mod.block(null, children, binaryen.auto),
        grad: local.grad,
        bwd,
      };
    }
  }

  const(ref: binaryen.ExpressionRef, info: binaryen.ConstInfo): Result {
    if (typeof info.value !== "number")
      throw Error(`Unsupported constant kind: ${typeof info.value}`);
    switch (info.type) {
      case binaryen.f64:
        return {
          fwd: this.mod.f64.const(info.value), // TODO: other types
          grad: this.fwdZeroF64,
          bwd: this.bwdGrad(ref),
        };
      default:
        throw Error(`Unsupported constant type: ${info.type}`);
    }
  }

  binary(ref: binaryen.ExpressionRef, info: binaryen.BinaryInfo): Result {
    const left = this.expr(info.left);
    const right = this.expr(info.right);
    const dx = left.bwd;
    const dy = right.bwd;
    const dz = this.bwdGrad(ref);
    switch (info.op) {
      case binaryen.AddFloat64: {
        this.bwd.push(
          this.mod.local.set(dx, this.mod.f64.add(this.get(dx), this.get(dz))),
          this.mod.local.set(dy, this.mod.f64.add(this.get(dy), this.get(dz))),
        );
        return {
          fwd: this.mod.f64.add(left.fwd, right.fwd),
          grad: this.fwdZeroF64,
          bwd: dz,
        };
      }
      case binaryen.SubFloat64: {
        this.bwd.push(
          this.mod.local.set(dx, this.mod.f64.add(this.get(dx), this.get(dz))),
          this.mod.local.set(dy, this.mod.f64.sub(this.get(dy), this.get(dz))),
        );
        return {
          fwd: this.mod.f64.sub(left.fwd, right.fwd),
          grad: this.fwdZeroF64,
          bwd: dz,
        };
      }
      case binaryen.MulFloat64: {
        this.bwd.push(
          this.mod.local.set(
            dx,
            this.mod.f64.add(
              this.get(dx),
              this.mod.f64.mul(this.get(dz), this.untape(info.right)),
            ),
          ),
          this.mod.local.set(
            dy,
            this.mod.f64.add(
              this.get(dy),
              this.mod.f64.mul(this.get(dz), this.untape(info.left)),
            ),
          ),
        );
        return {
          fwd: this.mod.f64.mul(left.fwd, right.fwd),
          grad: this.fwdZeroF64,
          bwd: dz,
        };
      }
      case binaryen.DivFloat64: {
        const dx1 = this.makeBwd(binaryen.f64);
        this.bwd.push(
          this.mod.local.set(dx, this.mod.f64.add(this.get(dx), this.get(dx1))),
          this.mod.local.set(
            dy,
            this.mod.f64.sub(
              this.get(dy),
              this.mod.f64.mul(
                this.mod.local.tee(
                  dx1,
                  this.mod.f64.div(this.get(dz), this.untape(info.right)),
                  binaryen.f64,
                ),
                this.untape(ref),
              ),
            ),
          ),
        );
        return {
          fwd: this.mod.f64.div(left.fwd, right.fwd),
          grad: this.fwdZeroF64,
          bwd: dz,
        };
      }
      default:
        throw Error(`Unsupported binary operation: ${info.op}`);
    }
  }

  structNew(ref: binaryen.ExpressionRef, info: util.StructNewInfo): Result {
    if (info.operands.length !== 0)
      throw Error("Struct initializer not supported");
    const gradType = this.types.type(info.type);
    const grad = this.makeFwd(gradType);
    const bwd = this.bwdGrad(ref);
    return {
      fwd: this.mod.block(
        null,
        [
          this.mod.local.set(
            grad,
            util.structNew(this.mod, [], util.typeGetHeapType(gradType)),
          ),
          util.structNew(this.mod, [], util.typeGetHeapType(info.type)),
        ],
        info.type,
      ),
      grad,
      bwd,
    };
  }

  arrayNew(ref: binaryen.ExpressionRef, info: util.ArrayNewInfo): Result {
    if (info.init !== 0) throw Error("Array initializer not supported");
    const size = this.expr(info.size);
    const gradType = this.types.type(info.type);
    const gradHeapType = util.typeGetHeapType(gradType);
    const grad = this.makeFwd(gradType);
    const bwd = this.bwdGrad(ref);
    if (util.heapTypeIsStruct(gradHeapType))
      return {
        fwd: this.mod.block(
          null,
          [
            this.mod.local.set(
              grad,
              util.structNew(this.mod, [], gradHeapType),
            ),
            util.arrayNew(
              this.mod,
              util.typeGetHeapType(info.type),
              size.fwd,
              0,
            ),
          ],
          info.type,
        ),
        grad,
        bwd,
      };
    else {
      const sizeVar = this.makeFwd(binaryen.i32);
      return {
        fwd: this.mod.block(
          null,
          [
            this.mod.local.set(
              grad,
              util.arrayNew(
                this.mod,
                gradHeapType,
                this.mod.local.tee(sizeVar, size.fwd, binaryen.i32),
                0,
              ),
            ),
            util.arrayNew(
              this.mod,
              util.typeGetHeapType(info.type),
              this.fwdGet(sizeVar),
              0,
            ),
          ],
          info.type,
        ),
        grad,
        bwd,
      };
    }
  }

  arrayGet(ref: binaryen.ExpressionRef, info: util.ArrayGetInfo): Result {
    const arr = this.expr(info.ref);
    const index = this.expr(info.index);
    const gradType = this.types.type(info.type);
    const bwd = this.bwdGrad(ref);
    if (becomesMutable(info.type)) {
      this.bwd.push(
        util.arraySet(
          this.mod,
          this.get(arr.bwd),
          this.untape(info.index),
          // TODO: `f32`
          this.mod.f64.add(
            util.arrayGet(
              this.mod,
              this.get(arr.bwd),
              this.untape(info.index),
              gradType,
            ),
            this.get(bwd),
          ),
        ),
      );
      return {
        fwd: util.arrayGet(this.mod, arr.fwd, index.fwd, info.type),
        grad: this.fwdZeroF64, // TODO: `f32`
        bwd,
      };
    } else if (gradType === unit)
      return {
        fwd: util.arrayGet(
          this.mod,
          arr.fwd,
          index.fwd,
          info.type,
          info.isSigned,
        ),
        grad: this.fwdVoid,
        bwd,
      };
    else {
      const indexVar = this.makeFwd(binaryen.i32);
      const grad = this.makeFwd(gradType);
      return {
        fwd: util.arrayGet(
          this.mod,
          arr.fwd, // this must be evaluated before we use `arr.grad` below
          this.mod.block(
            null,
            [
              this.mod.local.set(
                grad,
                util.arrayGet(
                  this.mod,
                  this.fwdGet(arr.grad),
                  this.mod.local.tee(indexVar, index.fwd, binaryen.i32),
                  gradType,
                ),
              ),
              this.fwdGet(indexVar),
            ],
            binaryen.i32,
          ),
          info.type,
        ),
        grad,
        bwd,
      };
    }
  }

  arraySet(ref: binaryen.ExpressionRef, info: util.ArraySetInfo): Result {
    const arr = this.expr(info.ref);
    const index = this.expr(info.index);
    const value = this.expr(info.value);
    const type = binaryen.getExpressionType(info.value);
    const gradType = this.types.type(type);
    const grad = this.fwdVoid;
    const bwd = this.bwdGrad(ref);
    if (becomesMutable(type)) {
      // TODO: other types
      this.bwd.push(
        util.arraySet(
          this.mod,
          this.get(arr.bwd),
          this.untape(info.index),
          this.mod.f64.const(0),
        ),
        this.mod.local.set(
          value.bwd,
          this.mod.f64.add(
            this.get(value.bwd),
            util.arrayGet(
              this.mod,
              this.get(arr.bwd),
              this.untape(info.index),
              gradType,
            ),
          ),
        ),
      );
      return {
        fwd: util.arraySet(this.mod, arr.fwd, index.fwd, value.fwd),
        grad,
        bwd,
      };
    } else if (gradType === unit)
      return {
        fwd: util.arraySet(this.mod, arr.fwd, index.fwd, value.fwd),
        grad,
        bwd,
      };
    else {
      const field = util.unwrap(this.tape.sets.get(ref));
      this.bwd.push(
        util.arraySet(
          this.mod,
          this.get(arr.bwd),
          this.untape(info.index),
          this.get(this.bwdFields[field]),
        ),
      );
      const indexVar = this.makeFwd(binaryen.i32);
      return {
        fwd: this.mod.block(null, [
          util.arraySet(
            this.mod,
            arr.fwd,
            this.mod.local.tee(indexVar, index.fwd, binaryen.i32),
            value.fwd,
          ),
          this.mod.local.set(
            this.fwdFields[field],
            util.arrayGet(
              this.mod,
              this.fwdGet(arr.grad),
              this.fwdGet(indexVar),
              gradType,
            ),
          ),
          util.arraySet(
            this.mod,
            this.fwdGet(arr.grad),
            this.fwdGet(indexVar),
            this.fwdGet(value.grad),
          ),
        ]),
        grad,
        bwd,
      };
    }
  }

  arrayLen(ref: binaryen.ExpressionRef, info: util.ArrayLenInfo): Result {
    return {
      fwd: util.arrayLen(this.mod, this.expr(info.ref).fwd),
      grad: this.fwdVoid,
      bwd: this.bwdVoid,
    };
  }

  expression(
    ref: binaryen.ExpressionRef,
    info: binaryen.ExpressionInfo,
  ): Result {
    switch (info.id) {
      case binaryen.BlockId:
        return this.block(ref, info as binaryen.BlockInfo);
      case binaryen.CallId:
        return this.call(ref, info as binaryen.CallInfo);
      case binaryen.LocalGetId:
        return this.localGet(ref, info as binaryen.LocalGetInfo);
      case binaryen.LocalSetId:
        return this.localSet(ref, info as binaryen.LocalSetInfo);
      case binaryen.ConstId:
        return this.const(ref, info as binaryen.ConstInfo);
      case binaryen.BinaryId:
        return this.binary(ref, info as binaryen.BinaryInfo);
      case binaryen.StructNewId:
        return this.structNew(ref, info as util.StructNewInfo);
      case binaryen.ArrayNewId:
        return this.arrayNew(ref, info as util.ArrayNewInfo);
      case binaryen.ArrayGetId:
        return this.arrayGet(ref, info as util.ArrayGetInfo);
      case binaryen.ArraySetId:
        return this.arraySet(ref, info as util.ArraySetInfo);
      case binaryen.ArrayLenId:
        return this.arrayLen(ref, info as util.ArrayLenInfo);
      default:
        throw Error(`Unsupported expression ID: ${info.id}`);
    }
  }

  expr(ref: binaryen.ExpressionRef): Result {
    const info = util.getExpressionInfo(ref);
    let { fwd, grad, bwd } = this.expression(ref, info);
    const field = this.tape.stores.get(ref);
    const gradField = this.tape.grads.get(ref);
    if (gradField === undefined) {
      if (field !== undefined) {
        const index = this.fwdFields[field];
        fwd = this.mod.local.tee(index, fwd, this.fwdVars[index]);
      }
      return { fwd, grad, bwd };
    } else {
      const index =
        field === undefined ? this.makeFwd(info.type) : this.fwdFields[field];
      const gradIndex = this.fwdFields[gradField];
      return {
        fwd: this.mod.block(
          null,
          [
            this.mod.local.set(index, fwd),
            this.mod.local.set(gradIndex, this.fwdGet(grad)),
            this.fwdGet(index),
          ],
          info.type,
        ),
        grad,
        bwd,
      };
    }
  }
}

export interface Gradient {
  fwd: binaryen.FunctionRef;
  bwd: binaryen.FunctionRef;
}

/**
 * Return the forward and backward passes added to `mod` for each preexisting
 * function.
 */
export const autodiff = (mod: binaryen.Module): Gradient[] => {
  const types = new Types();
  const tapes = makeTapes(types, mod);
  const infos = tapes.map((_, i) =>
    binaryen.getFunctionInfo(mod.getFunctionByIndex(i)),
  );
  const names = makeNames(infos.map(({ name }) => name));
  return infos.map((f, i) => {
    const tape = tapes[i];
    const { fwd: fwdName, bwd: bwdName } = util.unwrap(names.get(f.name));

    const ad = new Autodiff(types, mod, names, f, tape);
    const tapeVar = ad.bwdParams.length - 1;
    const { fwd: fwdBody, grad: fwdGrad, bwd: gradResults } = ad.expr(f.body);

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
          // `createType([])` is incompatible with `local.set`
          ad.results.length === 0 ? fwdBody : mod.local.set(out, fwdBody),
          util.tupleMake(mod, [
            ...ad.results.map((_, i) => mod.tuple.extract(ad.fwdGet(out), i)),
            ...ad.resultsGrad.map((_, i) =>
              mod.tuple.extract(ad.fwdGet(fwdGrad), i),
            ),
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
              util.structGet(mod, i, ad.get(tapeVar), ad.bwdVars[index]),
            ),
          ),
          ...(ad.resultsGrad.length === 0
            ? [] // `(tuple.make)` is not allowed
            : [
                mod.local.set(
                  gradResults,
                  util.tupleMake(
                    mod,
                    ad.resultsGrad.map((_, i) =>
                      ad.get(ad.paramsGrad.length + i),
                    ),
                  ),
                ),
              ]),
          ...ad.bwd.reverse(),
          ...(ad.paramsGrad.length === 0
            ? [] // `(tuple.make)` is not allowed
            : [
                util.tupleMake(
                  mod,
                  ad.paramsGrad.map((_, index) => ad.get(index)),
                ),
              ]),
        ],
        bwdResult,
      ),
    );

    return { fwd, bwd };
  });
};
