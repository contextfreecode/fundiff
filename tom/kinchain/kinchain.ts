type Arm = { angles: number[]; lengths: number[] };

type Transform = { angle: number; point: Point };

type Point = { x: number; y: number };

function forward({ angles, lengths }: Arm): Transform[] {
  let end = { angle: 0, point: { x: 0, y: 0 } } as Transform;
  return lengths.map(
    (length, index) => {
      const angle = angles[index] + end.angle;
      end = {
        angle,
        point: {
          x: end.point.x + length * Math.cos(angle),
          y: end.point.y + length * Math.sin(angle),
        },
      };
      return end;
    },
  );
}

export function main() {
  const field = document.getElementById("field") as HTMLTextAreaElement;
  if (!field.value.trim()) {
    field.value = defaultText;
  }
  const update = () => {
    const angles = parseRow(field);
    const lengths = parseLengths();
    const chain = forward({ angles, lengths });
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

function parseLine(line: string): number[] {
  const texts = line.replaceAll(/[^-+\d\.e]+/g, " ").trim().split(/\s+/);
  const vals = texts.map((text) => parseFloat(text));
  return vals;
}

function parseRow(field: HTMLTextAreaElement): number[] {
  const text: string = field.value;
  if (!text.trim()) {
    return [];
  }
  const index: number = field.selectionStart;
  return parseRowAt(text, index);
}

function parseRowAt(text: string, index: number): number[] {
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

function parseLengths(): number[] {
  const field = document.getElementById("field") as HTMLTextAreaElement;
  const lines = field.value.split("\n") as string[];
  return parseLine(lines[0]);
}

function render(chain: Transform[]) {
  const path = document.getElementById("path")!;
  const coords = chain.flatMap(({ point: { x, y } }) => [x, y]);
  const d = chain.length ? `M 0 0 L ${coords.join(" ")}` : "";
  path.setAttribute("d", d);
}

const defaultText = `
[1.  1.  0.5]
[ 1.5707964 -0.7853982 -0.7853982]
[ 1.5211604 -0.8410561 -0.8102162]
[ 1.4721947  -0.8976971  -0.83514845]
[ 1.4242857 -0.9551287 -0.8601375]
[ 1.37786   -1.0131428 -0.8851361]
[ 1.3333864 -1.0715146 -0.9101099]
[ 1.2913787 -1.1299994 -0.9350396]
[ 1.2523961  -1.1883271  -0.95992213]
[ 1.2170416 -1.246193  -0.9847699]
[ 1.1859533 -1.303247  -1.0096073]
[ 1.1597826 -1.3590806 -1.0344645]
[ 1.139155  -1.41322   -1.0593673]
[ 1.1246057 -1.4651313 -1.0843254]
[ 1.1164975 -1.5142484 -1.1093218]
[ 1.1149397 -1.5600291 -1.1343086]
[ 1.1197381 -1.6020302 -1.159213 ]
[ 1.1304063 -1.6399755 -1.183953 ]
[ 1.1462373 -1.6737956 -1.2084575]
[ 1.1664053 -1.7036253 -1.2326804]
[ 1.190059  -1.7297765 -1.2566102]
[ 1.2163796 -1.7527039 -1.2802706]
`.trim();

main();
