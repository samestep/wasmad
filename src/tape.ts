import binaryen from "binaryen";
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

  /** field indices for the tapes of all the function's call expressions */
  calls: Map<binaryen.ExpressionRef, util.BinaryenIndex>;

  /** how to load all the primal values needed in the backward pass */
  loads: Map<binaryen.ExpressionRef, Load>;
}

class Taper implements Block {
  fields: number;
  stores: Map<binaryen.ExpressionRef, util.BinaryenIndex>;
  calls: Map<binaryen.ExpressionRef, util.BinaryenIndex>;
  loads: Map<binaryen.ExpressionRef, Load>;

  /** current values for the function's variables, used to mimic SSA */
  vars: Value[];

  constructor(f: binaryen.FunctionInfo) {
    this.fields = 0;
    this.stores = new Map();
    this.calls = new Map();
    this.loads = new Map();

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
    if (typeof info.value !== "number") throw Error("Unsupported constant");
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
      default:
        throw Error("Unsupported expression");
    }
  }

  expr(ref: binaryen.ExpressionRef): Value {
    return this.expression(ref, binaryen.getExpressionInfo(ref));
  }
}

export interface Tape extends Block {
  /** the struct type for this function's tape */
  struct: util.BinaryenHeapType;
}

/** Return a tape type for every function. */
export const makeTapes = (mod: binaryen.Module): Tape[] => {
  const indices = util.funcIndicesByName(mod);
  const n = mod.getNumFunctions();
  const blocks: Block[] = [];
  const types = util.buildType(n, (builder) => {
    builder.createRecGroup(0, n);
    const temps: binaryen.Type[] = [];
    for (let i = 0; i < n; ++i)
      temps.push(builder.getTempRefType(builder.getTempHeapType(i), false));
    for (let i = 0; i < n; ++i) {
      const f = binaryen.getFunctionInfo(mod.getFunctionByIndex(i));
      const t = new Taper(f);
      t.expr(f.body);
      const { fields, stores, calls, loads } = t;
      blocks.push({ fields, stores, calls, loads });
      const struct: util.Field[] = [];
      for (const [ref, i] of stores)
        struct[i] = {
          type: binaryen.getExpressionType(ref),
          packedType: util.packedTypeNotPacked,
          mutable: false,
        };
      for (const [ref, i] of calls)
        struct[i] = {
          type: temps[
            util.unwrap(
              indices.get(
                (binaryen.getExpressionInfo(ref) as binaryen.CallInfo).target,
              ),
            )
          ],
          packedType: util.packedTypeNotPacked,
          mutable: false,
        };
      builder.setStructType(i, struct);
    }
  });
  return types.map((struct, i) => ({ ...blocks[i], struct }));
};
