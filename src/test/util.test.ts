import binaryen from "binaryen";
import { describe } from "vitest";
import * as util from "../util.js";
import { goldenfile } from "./util.js";

describe("buildType", () => {
  goldenfile("struct.wat", () => {
    const mod = new binaryen.Module();
    try {
      const type = util.typeFromHeapType(
        util.buildStructType([
          {
            type: binaryen.i32,
            packedType: util.packedTypeNotPacked,
            mutable: false,
          },
        ]),
        false,
      );
      mod.addFunction("foo", type, type, [], mod.local.get(0, type));
      return mod.emitText();
    } finally {
      mod.dispose();
    }
  });
});
