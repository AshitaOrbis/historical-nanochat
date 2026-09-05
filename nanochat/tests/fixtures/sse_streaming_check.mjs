// bq-1946: exercise the ACTUAL streaming SSE parser shipped inside
// nanochat/nanochat/ui.html (extracted verbatim, not reimplemented) against
// controlled chunk boundaries, the way the finding's own loopback
// fetch/ReadableStream reproduction did.
//
// Usage: node sse_streaming_check.mjs <path-to-ui.html>
// Exits 0 if every scenario produces the expected fullResponse, else exits 1
// with a description of the first failure on stderr.

import { readFileSync } from "node:fs";

const uiHtmlPath = process.argv[2];
if (!uiHtmlPath) {
  console.error("usage: node sse_streaming_check.mjs <path-to-ui.html>");
  process.exit(2);
}

const html = readFileSync(uiHtmlPath, "utf8");
const startMarker = "const reader = response.body.getReader();";
const endMarker = "const assistantMessageIndex = messages.length;";
const startIdx = html.indexOf(startMarker);
const endIdx = html.indexOf(endMarker, startIdx);
if (startIdx === -1 || endIdx === -1) {
  console.error("could not locate the streaming-fetch block in ui.html (markers not found)");
  process.exit(2);
}
const streamingCode = html.slice(startIdx, endIdx);

// Wrap the extracted block in an async function with the same free variables
// generateAssistantResponse() closes over in ui.html: response, assistantContent,
// chatContainer, and the fullResponse binding it declares and mutates itself.
const harnessSrc =
  "return (async function(response, assistantContent, chatContainer) {\n" +
  streamingCode +
  "\n  return fullResponse;\n})";
// eslint-disable-next-line no-new-func
const buildHarness = new Function(harnessSrc);
const runStreamingLoop = buildHarness();

function fakeResponseFromChunks(chunks) {
  let i = 0;
  return {
    body: {
      getReader() {
        return {
          read: async () => {
            if (i >= chunks.length) return { done: true, value: undefined };
            const value = chunks[i++];
            return { done: false, value };
          },
        };
      },
    },
  };
}

function fakeDom() {
  return {
    assistantContent: { textContent: "" },
    chatContainer: { scrollTop: 0, scrollHeight: 0 },
  };
}

const scenarios = [];

// Trigger 1 (from the finding): a valid `data: {"token": "Hello"}` SSE event
// split into two network reads INSIDE its JSON -- 11 bytes then 14 bytes,
// the exact split the finding's real loopback reproduction observed.
{
  const full = 'data: {"token": "Hello"}\n';
  const chunk1 = Buffer.from(full.slice(0, 11), "utf8");
  const chunk2 = Buffer.from(full.slice(11), "utf8");
  scenarios.push({
    name: "SSE event split mid-JSON across an 11-byte then 14-byte chunk",
    chunks: [chunk1, chunk2],
    expected: "Hello",
  });
}

// Trigger 2: an SSE event split across three reads to make sure buffering
// isn't accidentally a special-cased two-read fix.
{
  const full = 'data: {"token": "World"}\n';
  const chunk1 = Buffer.from(full.slice(0, 7), "utf8");
  const chunk2 = Buffer.from(full.slice(7, 15), "utf8");
  const chunk3 = Buffer.from(full.slice(15), "utf8");
  scenarios.push({
    name: "SSE event split across three reads",
    chunks: [chunk1, chunk2, chunk3],
    expected: "World",
  });
}

// Trigger 3: multiple complete events delivered in a single read (baseline;
// must still work after switching to buffered parsing).
{
  const full = 'data: {"token": "foo"}\ndata: {"token": "bar"}\n';
  scenarios.push({
    name: "two complete events in one read",
    chunks: [Buffer.from(full, "utf8")],
    expected: "foobar",
  });
}

// Trigger 4: a multibyte UTF-8 character (cafe with a 2-byte e-acute, U+00E9)
// split so the byte boundary falls INSIDE the 2-byte encoding.
{
  const full = 'data: {"token": "café"}\n';
  const bytes = Buffer.from(full, "utf8");
  // Find the 0xC3 lead byte of the 2-byte sequence and split right after it.
  const leadIdx = bytes.indexOf(0xc3);
  if (leadIdx === -1) throw new Error("test setup error: no multibyte lead byte found");
  const chunk1 = bytes.subarray(0, leadIdx + 1);
  const chunk2 = bytes.subarray(leadIdx + 1);
  scenarios.push({
    name: "multibyte UTF-8 character split across chunk boundary",
    chunks: [chunk1, chunk2],
    expected: "café",
  });
}

// Trigger 5 (documents the pre-existing "must not silently ignore" clause):
// a genuinely malformed COMPLETE data line must not throw out of the loop or
// silently corrupt subsequent, valid tokens.
{
  const full = 'data: {not valid json}\ndata: {"token": "ok"}\n';
  scenarios.push({
    name: "malformed complete event does not poison subsequent valid events",
    chunks: [Buffer.from(full, "utf8")],
    expected: "ok",
  });
}

let failures = 0;
for (const scenario of scenarios) {
  const { assistantContent, chatContainer } = fakeDom();
  const response = fakeResponseFromChunks(scenario.chunks);
  let actual;
  try {
    actual = await runStreamingLoop(response, assistantContent, chatContainer);
  } catch (e) {
    actual = `<threw: ${e}>`;
  }
  if (actual !== scenario.expected) {
    failures += 1;
    console.error(
      `FAIL: ${scenario.name}\n  expected=${JSON.stringify(scenario.expected)}\n  actual=${JSON.stringify(actual)}`
    );
  } else {
    console.log(`PASS: ${scenario.name}`);
  }
}

if (failures > 0) {
  console.error(`${failures}/${scenarios.length} scenario(s) failed`);
  process.exit(1);
}
console.log(`all ${scenarios.length} scenario(s) passed`);
process.exit(0);
