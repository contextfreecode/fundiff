function forward({ angles, lengths }) {
  let end = {
    angle: 0,
    point: {
      x: 0,
      y: 0,
    },
  };
  return lengths.map((length, index) => {
    const angle = angles[index] + end.angle;
    end = {
      angle,
      point: {
        x: end.point.x + length * Math.cos(angle),
        y: end.point.y + length * Math.sin(angle),
      },
    };
    return end;
  });
}
function main1() {
  const field = document.getElementById("field");
  const update = () => {
    const angles = parseRow(field);
    const lengths = parseLengths();
    const chain = forward({
      angles,
      lengths,
    });
    render(chain);
  };
  update();
  document.addEventListener("selectionchange", (event) => {
    console.log(event);
    if (document.activeElement == field) {
      update();
    }
  });
}
function parseLine(line) {
  const texts = line.replaceAll(/[^-+\d\.e]+/g, " ").trim().split(/\s+/);
  const vals = texts.map((text) => parseFloat(text));
  return vals;
}
function parseRow(field) {
  const text = field.value;
  if (!text.trim()) {
    return [];
  }
  const index = field.selectionStart;
  return parseRowAt(text, index);
}
function parseRowAt(text, index) {
  const begin = Math.max(0, text.lastIndexOf("\n", index - 1));
  let end = text.indexOf("\n", index);
  if (end < 0) {
    end = text.length;
  }
  if (begin <= 0) {
    return parseRowAt(text, end + 1);
  }
  const line = text.slice(begin, end);
  if (!line.trim()) {
    return parseRowAt(text, begin - 1);
  }
  return parseLine(line);
}
function parseLengths() {
  const field = document.getElementById("field");
  const lines = field.value.split("\n");
  return parseLine(lines[0]);
}
function render(chain) {
  const path = document.getElementById("path");
  const coords = chain.flatMap(({ point: { x, y } }) => [
    x,
    y,
  ]);
  const d = chain.length ? `M 0 0 L ${coords.join(" ")}` : "";
  path.setAttribute("d", d);
}
main1();
export { main1 as main };
