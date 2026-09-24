// Run with: node tests/js/paddle_sim.test.js
const assert = require("assert");
const { PaddleSim: S } = require("../../laya_axera/web/paddle.js");
let pass = 0;
const t = (name, fn) => { fn(); pass++; console.log("ok -", name); };
const run = (g, ms, now, controls = {}) => { for (let i = 0; i < ms / 10; i++) { now += 10; S.update(g, 10, now, controls); } return now; };

t("serves after a short delay, toward the receiver", () => {
  const g = S.create("easy", 0, 1);
  let now = run(g, 500, 0);
  assert.equal(g.ball, null);
  now = run(g, 600, now);
  assert.ok(g.ball, "served");
  assert.ok(g.ball.vy > 0, "first serve goes to the player (bottom)");
});

t("side rails reflect the ball", () => {
  const g = S.create("easy", 0, 1);
  run(g, 1000, 0);
  g.ball = { x: 0.5, y: 9, vx: -8, vy: 1, speed: Math.hypot(8, 1), fromY: 9, toY: S.PLAYER_Y };
  run(g, 200, 2000);
  assert.ok(g.ball.vx > 0 && g.ball.x >= S.BALL_R);
});

t("a paddle hit reverses the ball, speeds it up and counts the rally", () => {
  const g = S.create("easy", 0, 1);
  run(g, 1000, 0);
  g.you.x = 6;
  g.ball = { x: 6.5, y: 15.5, vx: 0, vy: 9, speed: 9, fromY: 9, toY: S.PLAYER_Y };
  run(g, 200, 2000);
  assert.ok(g.ball.vy < 0, "reversed");
  assert.ok(g.ball.speed > 9, "faster");
  assert.ok(g.ball.vx > 0, "hit right of centre adds rightward english");
  assert.equal(g.rally, 1);
});

t("a miss scores for the other side and the loser receives next", () => {
  const g = S.create("easy", 0, 1);
  run(g, 1000, 0);
  g.you.x = 1.3;
  g.ball = { x: 10, y: 15, vx: 0, vy: 9, speed: 9, fromY: 9, toY: S.PLAYER_Y };
  run(g, 800, 2000);
  assert.equal(g.score.ai, 1);
  assert.equal(g.ball, null);
  assert.equal(g.serveTo, "you");
});

t("first to WIN points ends the match", () => {
  const g = S.create("easy", 0, 1);
  let now = 1000;
  for (let i = 0; i < S.WIN; i++) {
    while (!g.ball) now = run(g, 50, now);
    g.ai.x = 1.3; g.ai.dir = 0;
    g.ball = { x: 10, y: 3, vx: 0, vy: -9, speed: 9, fromY: 9, toY: S.AI_Y };
    now = run(g, 800, now);
  }
  assert.equal(g.over, "you");
  assert.equal(g.score.you, S.WIN);
});

t("an AI move covers one step and stops; it never passes the predicted landing point", () => {
  const g = S.create("hell", 0, 1);
  g.ai.step = 2;
  S.applyAi(g, { action: "right", target: 11 });
  run(g, 3000, 0);
  assert.equal(g.ai.x, S.W / 2 + 2, "one step, then waits");
  S.applyAi(g, { action: "right", target: 8.5 });
  run(g, 3000, 3000);
  assert.equal(g.ai.x, 8.5, "stopped on the landing point");
  S.applyAi(g, { action: "right", target: 2 });
  run(g, 3000, 6000);
  assert.equal(g.ai.x, 10.5, "a move away from the landing point is not clipped");
  S.applyAi(g, { action: "right", target: 2 });
  run(g, 3000, 9000);
  assert.equal(g.ai.x, S.W - S.PADDLE_W / 2, "rail");
});

t("difficulty sets the AI paddle speed; the planned step spans speed x (think + latency)", () => {
  const moved = {}, steps = {};
  for (const lv of ["easy", "hard", "hell"]) {
    const g = S.create(lv, 0, 1);
    run(g, 1000, 0);
    steps[lv] = S.aiPayload(g, 100).ai.step;
    g.ai.step = 5;
    S.applyAi(g, { action: "left", target: 0 });
    run(g, 200, 1000);
    moved[lv] = S.W / 2 - g.ai.x;
  }
  assert.ok(moved.easy < moved.hard && moved.hard < moved.hell, JSON.stringify(moved));
  for (const lv of ["easy", "hard", "hell"]) {
    assert.equal(steps[lv], (S.LEVELS[lv].speed * (S.LEVELS[lv].think + 100)) / 1000);
  }
});

t("the player paddle follows the keys", () => {
  const g = S.create("easy", 0, 1);
  run(g, 200, 0, { dir: -1 });
  assert.ok(g.you.x < S.W / 2);
});

t("AI payload carries the court, the ball and a step sized to the decision interval", () => {
  const g = S.create("hard", 0, 1);
  run(g, 1000, 0);
  const p = S.aiPayload(g, 150);
  assert.deepEqual(Object.keys(p).sort(), ["ai", "ball", "court"]);
  assert.equal(p.ai.step, (S.LEVELS.hard.speed * (S.LEVELS.hard.think + 150)) / 1000);
  assert.ok(S.aiReady(g, 1000));
});
console.log(`\n${pass} passed`);
