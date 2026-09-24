"use strict";
// 2.5D drawing helpers shared by the game boards. Light comes from the top-left:
// top faces and rims are lit, extruded sides and floor shadows fall bottom-right.
const GFX = (() => {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function parse(color) {
    const c = color.trim();
    if (c.startsWith("#")) {
      const h = c.length === 4 ? c.slice(1).replace(/./g, (x) => x + x) : c.slice(1, 7);
      return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
    }
    const m = c.match(/[\d.]+/g);
    return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
  }

  // amount < 0 mixes toward black, > 0 toward white
  function shade(color, amount) {
    const [r, g, b] = parse(color);
    const t = amount < 0 ? 0 : 255;
    const k = Math.abs(amount);
    return `rgb(${[r, g, b].map((v) => Math.round(v + (t - v) * k)).join(",")})`;
  }

  function alpha(color, a) {
    const [r, g, b] = parse(color);
    return `rgba(${r},${g},${b},${a})`;
  }

  const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function rr(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, Math.max(0, h), Math.max(0, Math.min(r, w / 2, h / 2)));
  }

  // A raised block: floor shadow, darker extruded side, gradient top face, lit rim.
  function block(ctx, x, y, w, h, color, opt = {}) {
    const depth = opt.depth ?? Math.max(2, Math.round(Math.min(w, h) * 0.14));
    const r = opt.radius ?? Math.min(w, h) * 0.2;
    if (opt.shadow !== false) {
      ctx.fillStyle = "rgba(0,0,0,0.30)";
      rr(ctx, x + depth * 0.7, y + depth * 1.3, w, h, r);
      ctx.fill();
    }
    ctx.fillStyle = shade(color, -0.42);
    rr(ctx, x, y + depth, w, h - depth, r);
    ctx.fill();
    const top = ctx.createLinearGradient(x, y, x, y + h - depth);
    top.addColorStop(0, shade(color, 0.26));
    top.addColorStop(0.55, color);
    top.addColorStop(1, shade(color, -0.1));
    ctx.fillStyle = top;
    rr(ctx, x, y, w, h - depth, r);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.34)";
    ctx.lineWidth = Math.max(1, Math.min(w, h) * 0.05);
    ctx.beginPath();
    ctx.moveTo(x + r * 0.8, y + ctx.lineWidth);
    ctx.lineTo(x + w - r * 0.8, y + ctx.lineWidth);
    ctx.stroke();
  }

  // A glossy sphere with a specular spot.
  function sphere(ctx, cx, cy, r, color) {
    const g = ctx.createRadialGradient(cx - r * 0.38, cy - r * 0.42, r * 0.08, cx, cy, r);
    g.addColorStop(0, shade(color, 0.6));
    g.addColorStop(0.32, shade(color, 0.12));
    g.addColorStop(1, shade(color, -0.42));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.beginPath();
    ctx.ellipse(cx - r * 0.36, cy - r * 0.44, r * 0.2, r * 0.13, -0.6, 0, Math.PI * 2);
    ctx.fill();
  }

  function floorShadow(ctx, cx, cy, rx, ry, strength = 0.35) {
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rx);
    g.addColorStop(0, `rgba(0,0,0,${strength})`);
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(1, ry / rx);
    ctx.translate(-cx, -cy);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, rx, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // Vertical cylinder shading: dark edge, highlight band, body, dark edge.
  function cylinder(ctx, x, y, w, h, color) {
    const g = ctx.createLinearGradient(x, 0, x + w, 0);
    g.addColorStop(0, shade(color, -0.45));
    g.addColorStop(0.22, shade(color, 0.32));
    g.addColorStop(0.4, shade(color, 0.08));
    g.addColorStop(0.78, shade(color, -0.18));
    g.addColorStop(1, shade(color, -0.5));
    ctx.fillStyle = g;
    ctx.fillRect(x, y, w, h);
  }

  // A metallic capsule, e.g. a paddle.
  function capsule(ctx, x, y, w, h, color) {
    const g = ctx.createLinearGradient(0, y, 0, y + h);
    g.addColorStop(0, shade(color, 0.45));
    g.addColorStop(0.45, color);
    g.addColorStop(1, shade(color, -0.45));
    ctx.fillStyle = g;
    rr(ctx, x, y, w, h, h / 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = Math.max(1, h * 0.1);
    ctx.beginPath();
    ctx.moveTo(x + h * 0.6, y + h * 0.28);
    ctx.lineTo(x + w - h * 0.6, y + h * 0.28);
    ctx.stroke();
  }

  // Board floor: base color, soft top light and darkened edges.
  function floor(ctx, w, h, base) {
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, w, h);
    const light = ctx.createLinearGradient(0, 0, 0, h);
    light.addColorStop(0, "rgba(255,255,255,0.05)");
    light.addColorStop(1, "rgba(0,0,0,0.10)");
    ctx.fillStyle = light;
    ctx.fillRect(0, 0, w, h);
    vignette(ctx, w, h, 0.35);
  }

  function vignette(ctx, w, h, strength) {
    const g = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.35, w / 2, h / 2, Math.hypot(w, h) * 0.6);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, `rgba(0,0,0,${strength})`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  }

  function glow(ctx, cx, cy, r, color, a = 0.55) {
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    g.addColorStop(0, alpha(color, a));
    g.addColorStop(1, alpha(color, 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Short-lived debris for hits and breaks. Time-based so frame rate does not matter.
  class Particles {
    constructor() {
      this.items = [];
      this.last = 0;
    }
    burst(x, y, color, n = 14, speed = 220, size = 5) {
      for (let i = 0; i < n; i++) {
        const a = Math.random() * Math.PI * 2;
        const v = speed * (0.35 + Math.random() * 0.65);
        this.items.push({
          x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - speed * 0.3,
          life: 0, max: 0.45 + Math.random() * 0.4, color, size: size * (0.5 + Math.random() * 0.8),
        });
      }
    }
    get alive() {
      return this.items.length > 0;
    }
    draw(ctx, now = performance.now()) {
      const dt = this.last ? Math.min(0.05, (now - this.last) / 1000) : 0;
      this.last = now;
      this.items = this.items.filter((p) => (p.life += dt) < p.max);
      for (const p of this.items) {
        p.vy += 520 * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        const k = 1 - p.life / p.max;
        ctx.fillStyle = alpha(p.color, k);
        ctx.fillRect(p.x - (p.size * k) / 2, p.y - (p.size * k) / 2, p.size * k, p.size * k);
      }
      if (!this.items.length) this.last = 0;
    }
  }

  return { clamp, shade, alpha, token, rr, block, sphere, floorShadow, cylinder, capsule, floor, vignette, glow, Particles };
})();
