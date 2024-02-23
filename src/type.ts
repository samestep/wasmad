import binaryen from "binaryen";
import * as util from "./util.js";

/** empty tuple */
export const unit = binaryen.createType([]);

export const becomesMutable = (type: binaryen.Type): boolean => {
  switch (type) {
    case binaryen.f32:
    case binaryen.f64:
      return true;
    default:
      return false;
  }
};

const unsupported = (type: string): Error => Error(`unsupported type: ${type}`);

/** mapping from primal types to gradient types */
export class Types {
  private types: Map<binaryen.Type, binaryen.Type>;
  private heapTypes: Map<util.BinaryenHeapType, util.BinaryenHeapType>;

  constructor() {
    this.types = new Map();
    this.heapTypes = new Map();
  }

  private uncachedType(type: binaryen.Type): binaryen.Type {
    switch (type) {
      case binaryen.i32:
      case binaryen.i64:
        return unit;
      case binaryen.none:
      case binaryen.f32:
      case binaryen.f64:
      case binaryen.unreachable:
        return type;
      case binaryen.v128:
        throw unsupported("v128");
      case binaryen.funcref:
        throw unsupported("funcref");
      case binaryen.externref:
        throw unsupported("externref");
      case binaryen.anyref:
        throw unsupported("anyref");
      case binaryen.eqref:
        throw unsupported("eqref");
      case binaryen.i31ref:
        throw unsupported("i31ref");
      case util.structref:
        throw unsupported("structref");
      case util.arrayref:
        throw unsupported("arrayref");
      case binaryen.stringref:
        throw unsupported("stringref");
      case binaryen.stringview_wtf8:
        throw unsupported("stringview_wtf8");
      case binaryen.stringview_wtf16:
        throw unsupported("stringview_wtf16");
      case binaryen.stringview_iter:
        throw unsupported("stringview_iter");
      case util.nullref:
        throw unsupported("nullref");
      case util.nullexternref:
        throw unsupported("nullexternref");
      case util.nullfuncref:
        throw unsupported("nullfuncref");
      case binaryen.auto:
        throw unsupported("auto");
      default: {
        const types = binaryen.expandType(type);
        return types.length === 1
          ? util.typeFromHeapType(
              this.heapType(util.typeGetHeapType(type)),
              util.typeIsNullable(type),
            )
          : binaryen.createType(this.tuple(types));
      }
    }
  }

  private uncachedHeapType(
    heapType: util.BinaryenHeapType,
  ): util.BinaryenHeapType {
    if (util.heapTypeIsSignature(heapType))
      throw Error("unsupported heap type kind: signature");
    else if (util.heapTypeIsStruct(heapType))
      return util.buildStructType(
        util
          .structTypeGetFields(heapType)
          .map(({ type, mutable }) => ({
            type: this.type(type),
            packedType: util.packedTypeNotPacked,
            mutable: mutable || becomesMutable(type),
          }))
          .filter(({ type }) => type !== unit),
      );
    else if (util.heapTypeIsArray(heapType)) {
      const elementType = this.type(util.arrayTypeGetElementType(heapType));
      if (elementType === unit) return util.buildStructType([]);
      const isElementMutable = util.arrayTypeIsElementMutable(heapType);
      return util.buildArrayType(
        elementType,
        util.packedTypeNotPacked,
        isElementMutable || becomesMutable(elementType),
      );
    } else throw Error("unknown heap type kind");
  }

  type(type: binaryen.Type): binaryen.Type {
    return util.cached(type, this.types, () => this.uncachedType(type));
  }

  tuple(types: binaryen.Type[]): binaryen.Type[] {
    return types.map((type) => this.type(type)).filter((type) => type !== unit);
  }

  heapType(heapType: util.BinaryenHeapType): util.BinaryenHeapType {
    return util.cached(heapType, this.heapTypes, () =>
      this.uncachedHeapType(heapType),
    );
  }
}
