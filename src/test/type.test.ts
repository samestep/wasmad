import binaryen from "binaryen";
import { describe, expect, test } from "vitest";
import { Types, unit } from "../type.js";
import * as util from "../util.js";

describe("basic", () => {
  test("none", () => {
    expect(new Types().type(binaryen.none)).toBe(binaryen.none);
  });

  test("i32", () => {
    expect(new Types().type(binaryen.i32)).toBe(unit);
  });

  test("i64", () => {
    expect(new Types().type(binaryen.i64)).toBe(unit);
  });

  test("f32", () => {
    expect(new Types().type(binaryen.f32)).toBe(binaryen.f32);
  });

  test("f64", () => {
    expect(new Types().type(binaryen.f64)).toBe(binaryen.f64);
  });

  test("v128", () => {
    expect(() => new Types().type(binaryen.v128)).toThrow(
      "unsupported type: v128",
    );
  });

  test("funcref", () => {
    expect(() => new Types().type(binaryen.funcref)).toThrow(
      "unsupported type: funcref",
    );
  });

  test("externref", () => {
    expect(() => new Types().type(binaryen.externref)).toThrow(
      "unsupported type: externref",
    );
  });

  test("anyref", () => {
    expect(() => new Types().type(binaryen.anyref)).toThrow(
      "unsupported type: anyref",
    );
  });

  test("eqref", () => {
    expect(() => new Types().type(binaryen.eqref)).toThrow(
      "unsupported type: eqref",
    );
  });

  test("i31ref", () => {
    expect(() => new Types().type(binaryen.i31ref)).toThrow(
      "unsupported type: i31ref",
    );
  });

  test("structref", () => {
    expect(() => new Types().type(util.structref)).toThrow(
      "unsupported type: structref",
    );
  });

  test("arrayref", () => {
    expect(() => new Types().type(util.arrayref)).toThrow(
      "unsupported type: arrayref",
    );
  });

  test("stringref", () => {
    expect(() => new Types().type(binaryen.stringref)).toThrow(
      "unsupported type: stringref",
    );
  });

  test("stringview_wtf8", () => {
    expect(() => new Types().type(binaryen.stringview_wtf8)).toThrow(
      "unsupported type: stringview_wtf8",
    );
  });

  test("stringview_wtf16", () => {
    expect(() => new Types().type(binaryen.stringview_wtf16)).toThrow(
      "unsupported type: stringview_wtf16",
    );
  });

  test("stringview_iter", () => {
    expect(() => new Types().type(binaryen.stringview_iter)).toThrow(
      "unsupported type: stringview_iter",
    );
  });

  test("nullref", () => {
    expect(() => new Types().type(util.nullref)).toThrow(
      "unsupported type: nullref",
    );
  });

  test("nullexternref", () => {
    expect(() => new Types().type(util.nullexternref)).toThrow(
      "unsupported type: nullexternref",
    );
  });

  test("nullfuncref", () => {
    expect(() => new Types().type(util.nullfuncref)).toThrow(
      "unsupported type: nullfuncref",
    );
  });

  test("unreachable", () => {
    expect(new Types().type(binaryen.unreachable)).toBe(binaryen.unreachable);
  });

  test("auto", () => {
    expect(() => new Types().type(binaryen.auto)).toThrow(
      "unsupported type: auto",
    );
  });
});

describe("tuple", () => {
  test("()", () => {
    expect(new Types().type(unit)).toBe(unit);
  });

  test("(i32, f64)", () => {
    expect(
      new Types().type(binaryen.createType([binaryen.i32, binaryen.f64])),
    ).toBe(binaryen.f64);
  });

  test("(f32, i64)", () => {
    expect(
      new Types().type(binaryen.createType([binaryen.f32, binaryen.i64])),
    ).toBe(binaryen.f32);
  });

  test("(f64, i32, f32)", () => {
    expect(
      new Types().type(
        binaryen.createType([binaryen.f64, binaryen.i32, binaryen.f32]),
      ),
    ).toBe(binaryen.createType([binaryen.f64, binaryen.f32]));
  });
});

describe("heap", () => {
  const expectHeapType = (
    before: util.BinaryenHeapType,
    after: util.BinaryenHeapType,
  ) => {
    expect(new Types().heapType(before)).toBe(after);
  };

  test("(struct)", () => {
    const before = util.buildStructType([]);
    const after = before;
    expectHeapType(before, after);
  });

  test("(struct (field i32))", () => {
    const before = util.buildStructType([
      {
        type: binaryen.i32,
        packedType: util.packedTypeNotPacked,
        mutable: false,
      },
    ]);
    const after = util.buildStructType([]);
    expectHeapType(before, after);
  });

  test("(struct (field i64))", () => {
    const before = util.buildStructType([
      {
        type: binaryen.i64,
        packedType: util.packedTypeNotPacked,
        mutable: false,
      },
    ]);
    const after = util.buildStructType([]);
    expectHeapType(before, after);
  });

  test("(struct (field i32 f64))", () => {
    const before = util.buildStructType([
      {
        type: binaryen.i32,
        packedType: util.packedTypeNotPacked,
        mutable: false,
      },
      {
        type: binaryen.f64,
        packedType: util.packedTypeNotPacked,
        mutable: false,
      },
    ]);
    const after = util.buildStructType([
      {
        type: binaryen.f64,
        packedType: util.packedTypeNotPacked,
        mutable: true,
      },
    ]);
    expectHeapType(before, after);
  });

  test("(struct (field f32 i64))", () => {
    const before = util.buildStructType([
      {
        type: binaryen.f32,
        packedType: util.packedTypeNotPacked,
        mutable: false,
      },
      {
        type: binaryen.i64,
        packedType: util.packedTypeNotPacked,
        mutable: false,
      },
    ]);
    const after = util.buildStructType([
      {
        type: binaryen.f32,
        packedType: util.packedTypeNotPacked,
        mutable: true,
      },
    ]);
    expectHeapType(before, after);
  });

  test("(struct (field f64 i32 f32))", () => {
    const before = util.buildStructType([
      {
        type: binaryen.f64,
        packedType: util.packedTypeNotPacked,
        mutable: false,
      },
      {
        type: binaryen.i32,
        packedType: util.packedTypeNotPacked,
        mutable: false,
      },
      {
        type: binaryen.f32,
        packedType: util.packedTypeNotPacked,
        mutable: false,
      },
    ]);
    const after = util.buildStructType([
      {
        type: binaryen.f64,
        packedType: util.packedTypeNotPacked,
        mutable: true,
      },
      {
        type: binaryen.f32,
        packedType: util.packedTypeNotPacked,
        mutable: true,
      },
    ]);
    expectHeapType(before, after);
  });

  test("(array i32)", () => {
    const before = util.buildArrayType(
      binaryen.i32,
      util.packedTypeNotPacked,
      false,
    );
    const after = util.buildStructType([]);
    expectHeapType(before, after);
  });

  test("(array (mut i32))", () => {
    const before = util.buildArrayType(
      binaryen.i32,
      util.packedTypeNotPacked,
      true,
    );
    const after = util.buildStructType([]);
    expectHeapType(before, after);
  });

  test("(array f64)", () => {
    const before = util.buildArrayType(
      binaryen.f64,
      util.packedTypeNotPacked,
      false,
    );
    const after = util.buildArrayType(
      binaryen.f64,
      util.packedTypeNotPacked,
      true,
    );
    expectHeapType(before, after);
  });

  test("(array (mut f64))", () => {
    const before = util.buildArrayType(
      binaryen.f64,
      util.packedTypeNotPacked,
      true,
    );
    const after = before;
    expectHeapType(before, after);
  });

  describe("(type $vec (array (mut f64)))", () => {
    const heapType = util.buildArrayType(
      binaryen.f64,
      util.packedTypeNotPacked,
      true,
    );

    test("(struct (field f32 (ref $vec)))", () => {
      const ref = util.typeFromHeapType(heapType, false);
      const before = util.buildStructType([
        {
          type: binaryen.f32,
          packedType: util.packedTypeNotPacked,
          mutable: false,
        },
        { type: ref, packedType: util.packedTypeNotPacked, mutable: false },
      ]);
      const after = util.buildStructType([
        {
          type: binaryen.f32,
          packedType: util.packedTypeNotPacked,
          mutable: true,
        },
        { type: ref, packedType: util.packedTypeNotPacked, mutable: false },
      ]);
      expectHeapType(before, after);
    });

    test("(struct (field (mut f32) (ref null $vec)))", () => {
      const ref = util.typeFromHeapType(heapType, true);
      const before = util.buildStructType([
        {
          type: binaryen.f32,
          packedType: util.packedTypeNotPacked,
          mutable: true,
        },
        { type: ref, packedType: util.packedTypeNotPacked, mutable: false },
      ]);
      const after = before;
      expectHeapType(before, after);
    });

    test("(struct (field (mut (ref $vec)) (mut f64)))", () => {
      const ref = util.typeFromHeapType(heapType, false);
      const before = util.buildStructType([
        { type: ref, packedType: util.packedTypeNotPacked, mutable: true },
        {
          type: binaryen.f64,
          packedType: util.packedTypeNotPacked,
          mutable: true,
        },
      ]);
      const after = before;
      expectHeapType(before, after);
    });

    test("(struct (field (mut (ref null $vec)) f64))", () => {
      const ref = util.typeFromHeapType(heapType, true);
      const before = util.buildStructType([
        { type: ref, packedType: util.packedTypeNotPacked, mutable: true },
        {
          type: binaryen.f64,
          packedType: util.packedTypeNotPacked,
          mutable: false,
        },
      ]);
      const after = util.buildStructType([
        { type: ref, packedType: util.packedTypeNotPacked, mutable: true },
        {
          type: binaryen.f64,
          packedType: util.packedTypeNotPacked,
          mutable: true,
        },
      ]);
      expectHeapType(before, after);
    });

    test("(array (ref $vec))", () => {
      const ref = util.typeFromHeapType(heapType, false);
      const before = util.buildArrayType(ref, util.packedTypeNotPacked, false);
      const after = before;
      expectHeapType(before, after);
    });

    test("(array (ref null $vec))", () => {
      const ref = util.typeFromHeapType(heapType, true);
      const before = util.buildArrayType(ref, util.packedTypeNotPacked, false);
      const after = before;
      expectHeapType(before, after);
    });

    test("(array (mut (ref $vec)))", () => {
      const ref = util.typeFromHeapType(heapType, false);
      const before = util.buildArrayType(ref, util.packedTypeNotPacked, true);
      const after = before;
      expectHeapType(before, after);
    });

    test("(array (mut (ref null $vec)))", () => {
      const ref = util.typeFromHeapType(heapType, true);
      const before = util.buildArrayType(ref, util.packedTypeNotPacked, true);
      const after = before;
      expectHeapType(before, after);
    });
  });
});
