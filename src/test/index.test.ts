import binaryen from "binaryen";
import fs from "fs/promises";
import path from "path";
import url from "url";
import { expect, test } from "vitest";

test("subtraction", async () => {
  const dir = path.dirname(url.fileURLToPath(import.meta.url));
  const text = await fs.readFile(path.join(dir, "sub.wat"), "utf8");
  const module = binaryen.parseText(text);
  const binary = module.emitBinary();
  module.dispose();
  const compiled = await WebAssembly.compile(binary);
  const instance = await WebAssembly.instantiate(compiled, {});
  expect((instance.exports as any).sub(5, 3)).toBe(2);
});
