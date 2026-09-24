"use strict";
// Paddle duel on a table seen from the player's end. The table runs here in the
// page; only the AI paddle's moves come from the server, where Laya picks
// left / right / hold on the NPU. Difficulty limits how often the AI may decide
// and how fast its paddle slides, so a person can win.

const PaddleSim = (() => {
  const W = 12, L = 18;
  const AI_Y = 1.2, PLAYER_Y = 16.8;
  const PADDLE_W = 2.6, BALL_R = 0.28;
  const WIN = 5;
  const LEVELS = {
    easy: { label: "简单", think: 380, speed: 5.5 },
    hard: { label: "困难", think: 200, speed: 8.5 },
    hell: { label: "地狱", think: 110, speed: 11.5 },
  };
  const PLAYER_SPEED = 10.5;
  const SERVE_SPEED = 8.5, MAX_SPEED = 19, SPEEDUP = 1.045, ENGLISH = 3.2;
  const SERVE_DELAY = 800;

  function rand(g) {
    g.seed = (g.seed * 1103515245 + 12345) % 2147483648;
    return g.seed / 2147483648;
  }

  function create(level = "easy", now = 0, seed = 7) {
    return {
      level, W, L, seed,
      you: { x: W / 2, dir: 0 },
      ai: { x: W / 2, target: W / 2, step: 0 },
      ball: null,
      score: { you: 0, ai: 0 },
      rally: 0,
      over: null,
      events: [],
      serveAt: now + 900,
      serveTo: "you",
      aiBusy: false,
      aiNextAt: now + 500,
    };
  }

  function serve(g) {
    const angle = (rand(g) - 0.5) * 0.8;
    const toward = g.serveTo === "you" ? 1 : -1;
    g.ball = {
      x: W / 2, y: L / 2,
      vx: SERVE_SPEED * Math.sin(angle), vy: toward * SERVE_SPEED * Math.cos(angle),
      speed: SERVE_SPEED, fromY: L / 2, toY: toward > 0 ? PLAYER_Y : AI_Y,
    };
    g.rally = 0;
    g.events.push({ type: "serve" });
  }

  function slide(p, dir, speed, dt) {
    const half = PADDLE_W / 2;
    p.x = Math.max(half, Math.min(W - half, p.x + (dir * speed * dt) / 1000));
  }

  // The AI paddle glides to the spot its last decision picked, then waits.
  function glide(p, speed, dt) {
    const d = p.target - p.x, most = (speed * dt) / 1000;
    p.x = Math.abs(d) <= most ? p.target : p.x + Math.sign(d) * most;
  }

  // Same rule as laya_axera/paddle/planner.py `moved`: a move toward the predicted
  // landing point stops on it.
  function moved(x, action, step, target) {
    const half = PADDLE_W / 2;
    if (action === "hold") return x;
    const sign = action === "left" ? -1 : 1;
    let end = x + sign * step;
    if ((target - x) * sign > 0) end = sign > 0 ? Math.min(end, target) : Math.max(end, target);
    return Math.max(half, Math.min(W - half, end));
  }

  // Bounce off a paddle: reverse, add english from the contact point, speed up.
  function bounce(g, b, paddleX, toward) {
    const english = ((b.x - paddleX) / (PADDLE_W / 2)) * ENGLISH;
    b.speed = Math.min(MAX_SPEED, b.speed * SPEEDUP);
    let vx = b.vx + english;
    let vy = toward * Math.max(Math.abs(b.vy), b.speed * 0.45);
    const k = b.speed / Math.hypot(vx, vy);
    vx *= k; vy *= k;
    if (Math.abs(vy) < b.speed * 0.45) {
      vy = toward * b.speed * 0.45;
      vx = Math.sign(vx) * Math.sqrt(b.speed * b.speed - vy * vy);
    }
    b.vx = vx; b.vy = vy;
    b.fromY = b.y;
    b.toY = toward > 0 ? PLAYER_Y : AI_Y;
    g.rally += 1;
  }

  function point(g, winner, now) {
    g.score[winner] += 1;
    g.events.push({ type: "point", winner, x: g.ball.x, y: g.ball.y });
    g.ball = null;
    g.serveTo = winner === "you" ? "ai" : "you";
    g.serveAt = now + SERVE_DELAY;
    if (g.score[winner] >= WIN) g.over = winner;
  }

  // controls: { dir: -1 | 0 | 1 } for the human paddle.
  function update(g, dt, now, controls) {
    if (g.over) return;
    slide(g.you, controls ? controls.dir || 0 : 0, PLAYER_SPEED, dt);
    glide(g.ai, LEVELS[g.level].speed, dt);
    if (!g.ball) {
      if (now >= g.serveAt) serve(g);
      return;
    }
    const b = g.ball;
    let left = (b.speed * dt) / 1000;
    while (left > 0 && g.ball) {
      const step = Math.min(0.1, left);
      left -= step;
      const k = step / b.speed;
      const py = b.y;
      b.x += b.vx * k;
      b.y += b.vy * k;
      if (b.x < BALL_R) { b.x = 2 * BALL_R - b.x; b.vx = Math.abs(b.vx); }
      else if (b.x > W - BALL_R) { b.x = 2 * (W - BALL_R) - b.x; b.vx = -Math.abs(b.vx); }
      if (b.vy < 0 && py > AI_Y && b.y <= AI_Y && Math.abs(b.x - g.ai.x) <= PADDLE_W / 2 + BALL_R) {
        b.y = AI_Y;
        bounce(g, b, g.ai.x, 1);
        g.events.push({ type: "hit", side: "ai", x: b.x, y: b.y });
      } else if (b.vy > 0 && py < PLAYER_Y && b.y >= PLAYER_Y && Math.abs(b.x - g.you.x) <= PADDLE_W / 2 + BALL_R) {
        b.y = PLAYER_Y;
        bounce(g, b, g.you.x, -1);
        g.events.push({ type: "hit", side: "you", x: b.x, y: b.y });
      } else if (b.y < -0.6) {
        point(g, "you", now);
      } else if (b.y > L + 0.6) {
        point(g, "ai", now);
      }
    }
  }

  function aiReady(g, now) {
    return !g.over && !!g.ball && !g.aiBusy && now >= g.aiNextAt;
  }

  // latencyMs: the measured round trip, so a move covers the whole gap until the
  // next decision actually lands rather than just the configured think time.
  function aiPayload(g, latencyMs = 0) {
    const lv = LEVELS[g.level];
    const b = g.ball;
    g.ai.step = (lv.speed * (lv.think + latencyMs)) / 1000;
    return {
      court: { width: W, length: L, ai_y: AI_Y, player_y: PLAYER_Y },
      ball: { x: b.x, y: b.y, vx: b.vx, vy: b.vy, r: BALL_R },
      ai: { x: g.ai.x, w: PADDLE_W, step: g.ai.step },
    };
  }

  function applyAi(g, decision) {
    if (g.over || !["left", "right", "hold"].includes(decision.action)) return;
    g.ai.target = moved(g.ai.x, decision.action, g.ai.step, decision.target ?? g.ai.x);
  }

  return { W, L, AI_Y, PLAYER_Y, PADDLE_W, BALL_R, WIN, LEVELS, PLAYER_SPEED, create, update, aiReady, aiPayload, applyAi };
})();

const PaddleView = (() => {
  const S = PaddleSim;

  // Perspective of a flat table: screen width is proportional to 1/depth, and so
  // is the screen height below the horizon, so both follow 1/w linearly in t.
  function projector(canvas) {
    const Wc = canvas.width, Hc = canvas.height;
    const far = Wc * 0.5, near = Wc * 0.86, yFar = Hc * 0.14, yNear = Hc * 0.84;
    const widthAt = (t) => 1 / ((1 - t) / far + t / near);
    return {
      Wc, Hc, far, near, yFar, yNear,
      widthAt,
      pt(x, y) {
        const t = y / S.L;
        const w = widthAt(t);
        return { x: Wc / 2 + (x / S.W - 0.5) * w, y: yFar + ((yNear - yFar) * (w - far)) / (near - far), s: w / near };
      },
    };
  }

  function drawTable(ctx, P, dark) {
    const top = dark ? "#1f5e59" : "#2c7d75", bottom = dark ? "#26736c" : "#3a978d";
    const c0 = P.pt(0, 0), c1 = P.pt(S.W, 0), c2 = P.pt(S.W, S.L), c3 = P.pt(0, S.L);
    const thick = 22;
    for (const lx of [0.1, 0.9]) {
      const leg = P.pt(S.W * lx, S.L);
      const lg = ctx.createLinearGradient(leg.x - 8, 0, leg.x + 8, 0);
      lg.addColorStop(0, "#1b1e24"); lg.addColorStop(0.4, "#4a505a"); lg.addColorStop(1, "#16181d");
      ctx.fillStyle = lg;
      ctx.fillRect(leg.x - 8, leg.y + thick - 4, 16, 64);
      GFX.floorShadow(ctx, leg.x + 6, leg.y + thick + 62, 26, 7, 0.4);
    }
    ctx.fillStyle = GFX.shade(bottom, -0.45);
    ctx.beginPath();
    ctx.moveTo(c3.x, c3.y); ctx.lineTo(c2.x, c2.y); ctx.lineTo(c2.x, c2.y + thick); ctx.lineTo(c3.x, c3.y + thick);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = GFX.shade(bottom, -0.3);
    for (const [a, b2] of [[c0, c3], [c1, c2]]) {
      ctx.beginPath();
      ctx.moveTo(a.x, a.y); ctx.lineTo(b2.x, b2.y); ctx.lineTo(b2.x, b2.y + thick); ctx.lineTo(a.x, a.y + thick * a.s);
      ctx.closePath(); ctx.fill();
    }
    const g = ctx.createLinearGradient(0, c0.y, 0, c3.y);
    g.addColorStop(0, top); g.addColorStop(1, bottom);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(c0.x, c0.y); ctx.lineTo(c1.x, c1.y); ctx.lineTo(c2.x, c2.y); ctx.lineTo(c3.x, c3.y);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 3;
    ctx.stroke();
    const m0 = P.pt(S.W / 2, 0), m1 = P.pt(S.W / 2, S.L);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.beginPath(); ctx.moveTo(m0.x, m0.y); ctx.lineTo(m1.x, m1.y); ctx.stroke();
  }

  function drawNet(ctx, P) {
    const a = P.pt(-0.35, S.L / 2), b = P.pt(S.W + 0.35, S.L / 2);
    const h = 30 * a.s;
    ctx.fillStyle = "rgba(235,240,245,0.22)";
    ctx.fillRect(a.x, a.y - h, b.x - a.x, h);
    ctx.strokeStyle = "rgba(235,240,245,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = a.x; x <= b.x; x += 7) { ctx.moveTo(x, a.y - h); ctx.lineTo(x, a.y); }
    ctx.stroke();
    ctx.fillStyle = "#f4f6f8";
    ctx.fillRect(a.x - 2, a.y - h - 3, b.x - a.x + 4, 4);
    ctx.fillStyle = "#3b4048";
    ctx.fillRect(a.x - 6, a.y - h - 4, 5, h + 4);
    ctx.fillRect(b.x + 1, a.y - h - 4, 5, h + 4);
  }

  function drawPaddle(ctx, P, x, y, color) {
    const p = P.pt(x, y);
    const w = (S.PADDLE_W / S.W) * P.widthAt(y / S.L);
    const h = 16 * p.s + 6;
    GFX.floorShadow(ctx, p.x + 4, p.y + h * 0.7, w * 0.62, h * 0.55, 0.4);
    GFX.capsule(ctx, p.x - w / 2, p.y - h / 2, w, h, color);
  }

  function drawBall(ctx, P, b, color) {
    const p = P.pt(b.x, b.y);
    const r = Math.max(5, (S.BALL_R / S.W) * P.widthAt(b.y / S.L) * 1.25);
    const span = b.toY - b.fromY;
    const f = span ? Math.min(1, Math.max(0, (b.y - b.fromY) / span)) : 0;
    const lift = Math.sin(Math.PI * f) * 70 * p.s;
    GFX.floorShadow(ctx, p.x, p.y, r * 1.4, r * 0.55, 0.45 * (1 - f * 0.3));
    GFX.glow(ctx, p.x, p.y - lift, r * 2.6, color, 0.25);
    GFX.sphere(ctx, p.x, p.y - lift, r, color);
  }

  // g: a PaddleSim state or null for the idle table; view: { flashes, fx }
  function render(canvas, g, view) {
    const { flashes = [], fx = null } = view || {};
    const ctx = canvas.getContext("2d");
    const P = projector(canvas);
    const dark = effectiveTheme() === "dark";
    const bg = ctx.createLinearGradient(0, 0, 0, P.Hc);
    bg.addColorStop(0, dark ? "#0c0f14" : "#dfe4ea");
    bg.addColorStop(1, dark ? "#1a202a" : "#c3ccd6");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, P.Wc, P.Hc);
    GFX.glow(ctx, P.Wc / 2, P.Hc * 0.5, P.Wc * 0.62, dark ? "#6fb3ff" : "#ffffff", dark ? 0.08 : 0.35);
    drawTable(ctx, P, dark);
    const state = g || S.create("easy", 0);
    const you = GFX.token("--blue"), ai = GFX.token("--orange");
    const ballColor = dark ? "#fff3d6" : "#fffaf0";
    const ball = state.ball;
    drawPaddle(ctx, P, state.ai.x, S.AI_Y, ai);
    if (ball && ball.y < S.L / 2) drawBall(ctx, P, ball, ballColor);
    drawNet(ctx, P);
    if (ball && ball.y >= S.L / 2) drawBall(ctx, P, ball, ballColor);
    drawPaddle(ctx, P, state.you.x, S.PLAYER_Y, you);
    const now = performance.now();
    for (let i = flashes.length - 1; i >= 0; i--) {
      const f = flashes[i], k = (now - f.t0) / f.dur;
      if (k >= 1) { flashes.splice(i, 1); continue; }
      const p = P.pt(f.x, f.y);
      GFX.glow(ctx, p.x, p.y, 60 * p.s * (0.5 + k), f.color, 0.7 * (1 - k));
    }
    if (fx) fx.draw(ctx, now);
    return P;
  }

  return { render, projector };
})();

if (typeof module !== "undefined") module.exports = { PaddleSim, PaddleView };

if (typeof document !== "undefined") {
  (() => {
    const S = PaddleSim;
    const LABEL = { left: "◀ 左移", hold: "■ 不动", right: "▶ 右移" };
    const fx = new GFX.Particles();
    const flashes = [];
    let game = null, running = false, paused = false, gen = 0;
    let level = "easy", simNow = 0, lastFrame = 0, raf = 0;
    let decisions = 0, lastDecisionAt = 0;
    const held = new Set();

    function initRows() {
      const box = $("p-rows");
      box.innerHTML = "";
      for (const a of ["left", "hold", "right"]) {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.id = "pact-" + a;
        row.innerHTML = `<span class="opt">${LABEL[a]}</span>
          <span class="track"><span class="fill"></span></span>
          <span class="val">–</span>`;
        box.appendChild(row);
      }
    }

    function showDecision(d) {
      for (const a of ["left", "hold", "right"]) {
        const row = $("pact-" + a);
        const p = d.probabilities[a] || 0;
        row.classList.toggle("chosen", d.action === a);
        row.querySelector(".fill").style.width = (p * 100).toFixed(1) + "%";
        row.querySelector(".val").textContent = pct(p);
        row.querySelector(".opt").textContent = (d.action === a ? "✓ " : "") + LABEL[a];
      }
      pressFx("p-" + d.action);
      $("p-note").textContent = `AI：${LABEL[d.action]}，离预计落点 ${d.error.toFixed(1)} 格`;
      decisions += 1;
      const now = performance.now();
      $("p-s-dec").textContent = decisions;
      $("p-s-inf").innerHTML = d.inference_ms.toFixed(0) + "<small> ms</small>";
      if (lastDecisionAt) $("p-s-gap").innerHTML = (now - lastDecisionAt).toFixed(0) + "<small> ms</small>";
      lastDecisionAt = now;
    }

    function updateScore() {
      $("p-score-you").textContent = game ? game.score.you : 0;
      $("p-score-ai").textContent = game ? game.score.ai : 0;
      $("p-s-rally").textContent = game ? game.rally : 0;
    }

    let latency = 80;

    async function askAi(myGen) {
      game.aiBusy = true;
      const sent = performance.now();
      try {
        const d = await api("/api/paddle/decide", { model: state.current, ...S.aiPayload(game, latency) });
        latency = latency * 0.8 + (performance.now() - sent) * 0.2;
        if (myGen !== gen || !running) return;
        S.applyAi(game, d);
        showDecision(d);
      } catch (err) {
        if (myGen === gen) {
          showError("paddle-error", err.message);
          setPaused(true);
        }
      } finally {
        if (myGen === gen && game) {
          game.aiBusy = false;
          game.aiNextAt = simNow + S.LEVELS[game.level].think;
        }
      }
    }

    function handleEvents(canvas) {
      const P = PaddleView.projector(canvas);
      for (const e of game.events) {
        if (e.type === "hit") {
          const p = P.pt(e.x, e.y);
          flashes.push({ x: e.x, y: e.y, t0: performance.now(), dur: 220, color: "#fff2b0" });
          fx.burst(p.x, p.y, e.side === "you" ? GFX.token("--blue") : GFX.token("--orange"), 10, 200, 5);
        } else if (e.type === "point") {
          const p = P.pt(Math.min(S.W, Math.max(0, e.x)), Math.min(S.L, Math.max(0, e.y)));
          fx.burst(p.x, p.y, e.winner === "you" ? GFX.token("--blue") : GFX.token("--orange"), 24, 300, 6);
        }
      }
      game.events.length = 0;
      updateScore();
    }

    function heldDir() {
      const l = held.has("left"), r = held.has("right");
      return l && !r ? -1 : r && !l ? 1 : 0;
    }

    function frame(t) {
      raf = 0;
      if ($("view-paddle").hidden) {
        if (running && !paused) setPaused(true);
        lastFrame = 0;
        return;
      }
      const dt = lastFrame ? Math.min(40, t - lastFrame) : 16;
      lastFrame = t;
      const canvas = $("pboard");
      if (running && !paused) {
        simNow += dt;
        S.update(game, dt, simNow, { dir: heldDir() });
        if (S.aiReady(game, simNow)) askAi(gen);
        handleEvents(canvas);
        if (game.over) finish();
      }
      PaddleView.render(canvas, game, { flashes, fx });
      if ((running && !paused) || fx.alive || flashes.length) raf = requestAnimationFrame(frame);
    }

    function kick() {
      if (!raf) raf = requestAnimationFrame(frame);
    }

    function finish() {
      running = false;
      $("paddle-toggle").disabled = true;
      const won = game.over === "you";
      $("p-overlay-title").textContent = won ? "你赢了" : "AI 赢了";
      $("p-overlay-sub").textContent =
        `难度：${S.LEVELS[game.level].label}　比分 ${game.score.you} : ${game.score.ai}。点「开始对战」再来一局。`;
      $("p-overlay").hidden = false;
      $("paddle-new").textContent = "开始对战";
    }

    function setPaused(p) {
      paused = p;
      $("paddle-toggle").textContent = p ? "继续" : "暂停";
      if (!p) kick();
    }

    function start() {
      hide("paddle-error");
      gen += 1;
      simNow = 0;
      game = S.create(level, simNow, Math.floor(Math.random() * 1e9));
      running = true;
      decisions = 0;
      lastDecisionAt = 0;
      held.clear();
      $("p-overlay").hidden = true;
      $("paddle-toggle").disabled = false;
      $("paddle-new").textContent = "重新开始";
      $("p-s-dec").textContent = "0";
      $("p-s-inf").innerHTML = "–<small> ms</small>";
      $("p-s-gap").innerHTML = "–<small> ms</small>";
      $("p-note").textContent = "–";
      initRows();
      updateScore();
      setPaused(false);
      refreshInfo(true);
    }

    for (const b of document.querySelectorAll("#p-level button")) {
      b.onclick = () => {
        level = b.dataset.level;
        for (const o of document.querySelectorAll("#p-level button")) o.classList.toggle("active", o === b);
        if (game) game.level = level;
        if (!running) kick();
      };
    }
    $("paddle-new").onclick = start;
    $("paddle-toggle").onclick = () => setPaused(!paused);
    $("nav-paddle").addEventListener("click", () => {
      lastFrame = 0;
      kick();
    });

    document.addEventListener("keydown", (e) => {
      if ($("view-paddle").hidden || e.target.matches("textarea, input, select")) return;
      const k = KEYMAP[e.code];
      if (k !== "left" && k !== "right") return;
      held.add(k);
      e.preventDefault();
    });
    document.addEventListener("keyup", (e) => {
      const k = KEYMAP[e.code];
      if (k === "left" || k === "right") held.delete(k);
    });
    window.addEventListener("blur", () => held.clear());

    initRows();
    updateScore();
  })();
}
