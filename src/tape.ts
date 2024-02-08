import binaryen from "binaryen";
import * as util from "./util.js";

enum ValueKind {
  Void,
  Default,
  Expression,
}

type Value =
  | { kind: ValueKind.Void }
  | { kind: ValueKind.Default }
  | { kind: ValueKind.Expression; ref: binaryen.ExpressionRef };

class Taper {
  vars: Value[];
  need: Map<binaryen.ExpressionRef, Value>;

  constructor(f: binaryen.FunctionInfo) {
    this.vars = [...binaryen.expandType(f.params), ...f.vars].map(() => {
      return { kind: ValueKind.Default };
    });
    this.need = new Map();
  }

  mark(ref: binaryen.ExpressionRef, value: Value): Value {
    this.need.set(ref, value);
    return value;
  }

  save(ref: binaryen.ExpressionRef): Value {
    return this.mark(ref, this.expr(ref));
  }

  block(ref: binaryen.ExpressionRef, info: binaryen.BlockInfo): Value {
    let value: Value = { kind: ValueKind.Void };
    for (const child of info.children) value = this.expr(child);
    return value;
  }

  localGet(ref: binaryen.ExpressionRef, info: binaryen.LocalGetInfo): Value {
    return this.vars[info.index];
  }

  localSet(ref: binaryen.ExpressionRef, info: binaryen.LocalSetInfo): Value {
    const value = this.expr(info.value);
    this.vars[info.index] = value;
    return info.isTee ? value : { kind: ValueKind.Void };
  }

  const(ref: binaryen.ExpressionRef, info: binaryen.ConstInfo): Value {
    return { kind: ValueKind.Expression, ref };
  }

  binary(ref: binaryen.ExpressionRef, info: binaryen.BinaryInfo): Value {
    const value = { kind: ValueKind.Expression, ref };
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

interface Exprs {
  /** the expression that should be saved for each field in the struct */
  fwd: binaryen.ExpressionRef[];

  /** the struct field that should be used to retrieve each needed expression */
  bwd: Map<binaryen.ExpressionRef, util.BinaryenIndex>;
}

export interface Tape extends Exprs {
  /** the struct type for this function's tape */
  struct: util.BinaryenHeapType;
}

/** Return a tape type for every function. */
export const tape = (mod: binaryen.Module): Tape[] => {
  const n = mod.getNumFunctions();
  const exprs: Exprs[] = [];
  const types = util.buildType(n, (builder) => {
    for (let i = 0; i < n; ++i) {
      const f = binaryen.getFunctionInfo(mod.getFunctionByIndex(i));
      const t = new Taper(f);
      t.expr(f.body);
      const fields = new Map<binaryen.ExpressionRef, util.BinaryenIndex>();
      const bwd = new Map<binaryen.ExpressionRef, util.BinaryenIndex>();
      for (const [ref, value] of t.need) {
        if (value.kind === ValueKind.Expression) {
          let j = fields.get(value.ref);
          if (j === undefined) {
            j = fields.size;
            fields.set(value.ref, j);
          }
          bwd.set(ref, j);
        }
      }
      const fwd = [...fields.keys()];
      exprs.push({ fwd, bwd });
      builder.setStructType(
        i,
        fwd.map((ref) => ({
          type: binaryen.getExpressionType(ref),
          packedType: util.packedTypeNotPacked,
          mutable: false,
        })),
      );
    }
  });
  return types.map((struct, i) => ({ ...exprs[i], struct }));
};
