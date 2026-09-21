import { Mesh, Program, Renderer, Triangle, Vec2 } from "ogl";
import { useEffect, useRef } from "react";

import { fragment, vertex } from "./darkveil-shaders";

interface DarkVeilProps {
  hueShift?: number;
  speed?: number;
  warpAmount?: number;
  /** Render below native resolution — the shader is soft, so 0.5 is invisible and far cheaper. */
  resolutionScale?: number;
  /** Dim the canvas so foreground text keeps its contrast. */
  opacity?: number;
}

/**
 * Animated WebGL background. Improvements over v1: it pauses when the tab is
 * hidden, renders a single still frame under prefers-reduced-motion, renders at
 * reduced resolution, and releases the GL context on unmount.
 */
export function DarkVeil({
  hueShift = 0,
  speed = 0.35,
  warpAmount = 0.04,
  resolutionScale = 0.5,
  opacity = 0.55,
}: DarkVeilProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let renderer: Renderer;
    try {
      renderer = new Renderer({ canvas, dpr: Math.min(window.devicePixelRatio, 1.5), alpha: false });
    } catch {
      return; // no WebGL — the CSS gradient behind the canvas still shows
    }

    const gl = renderer.gl;
    const program = new Program(gl, {
      vertex,
      fragment,
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new Vec2() },
        uHueShift: { value: hueShift },
        uNoise: { value: 0 },
        uScan: { value: 0 },
        uScanFreq: { value: 0 },
        uWarp: { value: warpAmount },
      },
    });
    const mesh = new Mesh(gl, { geometry: new Triangle(gl), program });

    const resize = () => {
      const { innerWidth: w, innerHeight: h } = window;
      renderer.setSize(w * resolutionScale, h * resolutionScale);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      program.uniforms.uResolution.value.set(w * resolutionScale, h * resolutionScale);
    };
    resize();
    window.addEventListener("resize", resize);

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const start = performance.now();
    let frame = 0;

    const draw = () => {
      program.uniforms.uTime.value = ((performance.now() - start) / 1000) * speed;
      renderer.render({ scene: mesh });
    };
    const loop = () => {
      draw();
      frame = requestAnimationFrame(loop);
    };
    const onVisibility = () => {
      cancelAnimationFrame(frame);
      if (!document.hidden && !reducedMotion) loop();
    };

    if (reducedMotion) draw();
    else loop();
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", onVisibility);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    };
  }, [hueShift, speed, warpAmount, resolutionScale]);

  return (
    <div
      aria-hidden
      className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,#1b1640_0%,#07070c_60%)]"
    >
      <canvas ref={canvasRef} className="block" style={{ opacity }} />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-ink/30 to-ink/80" />
    </div>
  );
}
