import binaryen from "binaryen";
import { Types, becomesMutable, unit } from "./type.js";
import * as util from "./util.js";

enum ValueKind {
  /** this value is a parameter that has not been read yet */
  Param,

  /** this value is actually no values */
  Void,

  /** this value is the default for its type */
  Const,

  /** this value was computed by the given expression */
  Expression,
}

type Value =
  | { kind: ValueKind.Param }
  | { kind: ValueKind.Void }
  | { kind: ValueKind.Const; value: number }
  | { kind: ValueKind.Expression; ref: binaryen.ExpressionRef };

export enum LoadKind {
  /** constant */
  Const,

  /** field on the tape struct */
  Field,
}

export type Load =
  | { kind: LoadKind.Const; value: number }
  | { kind: LoadKind.Field; index: number };

export interface Block {
  /** the number of fields in the struct */
  fields: number;

  /** field indices for expressions that need to be teed in the forward pass */
  stores: Map<binaryen.ExpressionRef, util.BinaryenIndex>;

  /** field indices for gradients that need to be saved in the forward pass */
  grads: Map<binaryen.ExpressionRef, util.BinaryenIndex>;

  /** field indices for overwritten gradients from `array.set` expressions */
  sets: Map<binaryen.ExpressionRef, util.BinaryenIndex>;

  /** field indices for the tapes of all the function's call expressions */
  calls: Map<binaryen.ExpressionRef, util.BinaryenIndex>;

  /** how to load all the primal values needed in the backward pass */
  loads: Map<binaryen.ExpressionRef, Load>;

  /** how to load all the gradient values needed in the backward pass */
  gradLoads: Map<binaryen.ExpressionRef, Load>;
}

class Taper implements Block {
  fields: number;
  stores: Map<binaryen.ExpressionRef, util.BinaryenIndex>;
  grads: Map<binaryen.ExpressionRef, util.BinaryenIndex>;
  sets: Map<binaryen.ExpressionRef, util.BinaryenIndex>;
  calls: Map<binaryen.ExpressionRef, util.BinaryenIndex>;
  loads: Map<binaryen.ExpressionRef, Load>;
  gradLoads: Map<binaryen.ExpressionRef, Load>;

  /** current values for the function's variables, used to mimic SSA */
  vars: Value[];

  constructor(
    private types: Types,
    f: binaryen.FunctionInfo,
  ) {
    this.fields = 0;
    this.stores = new Map();
    this.grads = new Map();
    this.sets = new Map();
    this.calls = new Map();
    this.loads = new Map();
    this.gradLoads = new Map();

    const params = binaryen.expandType(f.params);
    this.vars = [
      ...params.map((): Value => ({ kind: ValueKind.Param })),
      ...f.vars.map((): Value => ({ kind: ValueKind.Const, value: 0 })),
    ];
  }

  mark(ref: binaryen.ExpressionRef, value: Value): Value {
    switch (value.kind) {
      case ValueKind.Param:
        throw Error("Parameter value should have been set at first read");
      case ValueKind.Void:
        throw Error("Void value should not be needed");
      case ValueKind.Const:
        this.loads.set(ref, { kind: LoadKind.Const, value: value.value });
        return value;
      case ValueKind.Expression: {
        let i = this.stores.get(value.ref);
        if (i === undefined) {
          i = this.fields++;
          this.stores.set(value.ref, i);
        }
        this.loads.set(ref, { kind: LoadKind.Field, index: i });
        return value;
      }
    }
  }

  // TODO: track gradient values just like primal values
  markGrad(ref: binaryen.ExpressionRef): void {
    let i = this.grads.get(ref);
    if (i === undefined) {
      i = this.fields++;
      this.grads.set(ref, i);
    }
    this.gradLoads.set(ref, { kind: LoadKind.Field, index: i });
  }

  save(ref: binaryen.ExpressionRef): Value {
    return this.mark(ref, this.expr(ref));
  }

  block(ref: binaryen.ExpressionRef, info: binaryen.BlockInfo): Value {
    let value: Value = { kind: ValueKind.Void };
    for (const child of info.children) value = this.expr(child);
    return value;
  }

  call(ref: binaryen.ExpressionRef, info: binaryen.CallInfo): Value {
    if (info.isReturn) throw Error("Tail call not supported");
    for (const operand of info.operands) this.expr(operand);
    this.calls.set(ref, this.fields++);
    return { kind: ValueKind.Expression, ref };
  }

  localGet(ref: binaryen.ExpressionRef, info: binaryen.LocalGetInfo): Value {
    let value = this.vars[info.index];
    if (value.kind === ValueKind.Param) {
      value = { kind: ValueKind.Expression, ref };
      this.vars[info.index] = value;
    }
    return value;
  }

  localSet(ref: binaryen.ExpressionRef, info: binaryen.LocalSetInfo): Value {
    const value = this.expr(info.value);
    this.vars[info.index] = value;
    return info.isTee ? value : { kind: ValueKind.Void };
  }

  const(ref: binaryen.ExpressionRef, info: binaryen.ConstInfo): Value {
    if (typeof info.value !== "number")
      throw Error(`Unsupported constant kind: ${typeof info.value}`);
    return { kind: ValueKind.Const, value: info.value };
  }

  binary(ref: binaryen.ExpressionRef, info: binaryen.BinaryInfo): Value {
    const value: Value = { kind: ValueKind.Expression, ref };
    switch (info.op) {
      case binaryen.MulFloat64:
        this.save(info.left);
        this.save(info.right);
        return value;
      case binaryen.DivFloat64:
        this.expr(info.left);
        this.save(info.right);
        return this.mark(ref, value);
      default:
        this.expr(info.left);
        this.expr(info.right);
        return value;
    }
  }

  structNew(ref: binaryen.ExpressionRef, info: util.StructNewInfo): Value {
    if (info.operands.length !== 0)
      throw Error("Struct initializer not supported");
    return { kind: ValueKind.Expression, ref };
  }

  arrayNew(ref: binaryen.ExpressionRef, info: util.ArrayNewInfo): Value {
    if (info.init !== 0) throw Error("Array initializer not supported");
    this.expr(info.size);
    return { kind: ValueKind.Expression, ref };
  }

  arrayGet(ref: binaryen.ExpressionRef, info: util.ArrayGetInfo): Value {
    this.expr(info.ref);
    if (becomesMutable(info.type)) {
      this.markGrad(info.ref);
      this.save(info.index);
    } else this.expr(info.index);
    return { kind: ValueKind.Expression, ref };
  }

  arraySet(ref: binaryen.ExpressionRef, info: util.ArraySetInfo): Value {
    this.expr(info.ref);
    this.save(info.index); // TODO: don't save when element gradient is unit
    this.expr(info.value);
    if (this.types.type(binaryen.getExpressionType(info.value)) !== unit) {
      this.markGrad(info.ref);
      this.markGrad(info.value);
      this.sets.set(ref, this.fields++); // TODO: only save when element is ref
    }
    return { kind: ValueKind.Expression, ref };
  }

  arrayLen(ref: binaryen.ExpressionRef, info: util.ArrayLenInfo): Value {
    this.expr(info.ref);
    return { kind: ValueKind.Expression, ref };
  }

  expression(
    ref: binaryen.ExpressionRef,
    info: binaryen.ExpressionInfo,
  ): Value {
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

  expr(ref: binaryen.ExpressionRef): Value {
    return this.expression(ref, util.getExpressionInfo(ref));
  }
}

export interface Tape extends Block {
  /** the struct type for this function's tape */
  struct: util.BinaryenHeapType;
}

/** Return a tape type for every function. */
export const makeTapes = (types: Types, mod: binaryen.Module): Tape[] => {
  const indices = util.funcIndicesByName(mod);
  const n = mod.getNumFunctions();
  const blocks: Block[] = [];
  const structs = util.buildTypes(n, (builder) => {
    builder.createRecGroup(0, n);
    const temps: binaryen.Type[] = [];
    for (let i = 0; i < n; ++i)
      temps.push(builder.getTempRefType(builder.getTempHeapType(i), false));
    for (let i = 0; i < n; ++i) {
      const f = binaryen.getFunctionInfo(mod.getFunctionByIndex(i));
      const t = new Taper(types, f);
      t.expr(f.body);
      const { fields, stores, grads, sets, calls, loads, gradLoads } = t;
      blocks.push({ fields, sets, stores, grads, calls, loads, gradLoads });
      const struct: util.Field[] = [];
      for (const [ref, i] of stores)
        struct[i] = {
          type: binaryen.getExpressionType(ref),
          packedType: util.packedTypeNotPacked,
          mutable: false,
        };
      for (const [ref, i] of grads)
        struct[i] = {
          type: types.type(binaryen.getExpressionType(ref)),
          packedType: util.packedTypeNotPacked,
          mutable: false,
        };
      for (const [ref, i] of sets)
        struct[i] = {
          type: types.type(
            binaryen.getExpressionType(
              (util.getExpressionInfo(ref) as util.ArraySetInfo).value,
            ),
          ),
          packedType: util.packedTypeNotPacked,
          mutable: false,
        };
      for (const [ref, i] of calls)
        struct[i] = {
          type: temps[
            util.unwrap(
              indices.get(
                (util.getExpressionInfo(ref) as binaryen.CallInfo).target,
              ),
            )
          ],
          packedType: util.packedTypeNotPacked,
          mutable: false,
        };
      builder.setStructType(i, struct);
    }
  });
  return structs.map((struct, i) => ({ ...blocks[i], struct }));
};
