type Arm = {angles: number[], lengths: number[]};

type Frame = {angle: number, point: Point};

type Point = {x: number, y: number};

function forward({angles, lengths}: Arm) {
  return lengths.reduce(
    (chain, length, index) => {
      const end = chain.slice(-1)[0];
      const angle = angles[index] + end.angle;
      const next =
        {angle, point: {x: end.point.x + length * Math.cos(angle)}} as Frame;
      return chain.concat([next]);
    },
    [{angle: 0, point: {x: 0, y: 0}} as Frame],
  );
}

function main() {
  console.log("hi");
}

function render() {
  document.getElementById("here");
}

main()
