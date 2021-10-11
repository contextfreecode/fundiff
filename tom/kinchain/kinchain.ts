type Arm = { angles: number[]; lengths: number[] };

type Transform = { angle: number; point: Point };

type Point = { x: number; y: number };

type Seq = { angleFrames: number[][]; lengths: number[] };

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
  const seq = parseSeq();
  const chain = forward({ angles: seq.angleFrames[0], lengths: seq.lengths });
  console.log(chain);
  render();
}

function parseSeq(): Seq {
  const field = document.getElementById("field") as HTMLTextAreaElement;
  const lines = field.value.split("\n") as string[];
  const rows = lines.map((line) => {
    const texts = line.replaceAll(/[^-+\d\.e]+/g, " ").trim().split(/\s+/);
    const vals = texts.map((text) => parseFloat(text));
    return vals;
  })
  return {angleFrames: rows.slice(1), lengths: rows[0]};
}

function render() {
  const path = document.getElementById("path");
  console.log(path);
}

main();
