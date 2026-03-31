import { useEffect, useRef } from "react"
import * as THREE from "three"

export default function AsteroidScene() {
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const w = mount.clientWidth
    const h = mount.clientHeight

    // ── Renderer ──────────────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setSize(w, h)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.2
    mount.appendChild(renderer.domElement)

    // ── Scene & Camera ────────────────────────────────────────────────────────
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x030712)
    scene.fog = new THREE.FogExp2(0x030712, 0.012)

    const camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 500)
    camera.position.set(0, 0, 8)

    // ── Stars ─────────────────────────────────────────────────────────────────
    const starCount = 3000
    const starPositions = new Float32Array(starCount * 3)
    const starSizes = new Float32Array(starCount)
    for (let i = 0; i < starCount; i++) {
      starPositions[i * 3]     = (Math.random() - 0.5) * 300
      starPositions[i * 3 + 1] = (Math.random() - 0.5) * 300
      starPositions[i * 3 + 2] = (Math.random() - 0.5) * 300
      starSizes[i] = Math.random() * 2 + 0.5
    }
    const starsGeo = new THREE.BufferGeometry()
    starsGeo.setAttribute("position", new THREE.BufferAttribute(starPositions, 3))
    starsGeo.setAttribute("size", new THREE.BufferAttribute(starSizes, 1))
    const starsMat = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.12,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.8,
    })
    const stars = new THREE.Points(starsGeo, starsMat)
    scene.add(stars)

    // ── Main Asteroid ─────────────────────────────────────────────────────────
    const asteroidGeo = new THREE.IcosahedronGeometry(2.2, 5)
    const posAttr = asteroidGeo.getAttribute("position") as THREE.BufferAttribute
    for (let i = 0; i < posAttr.count; i++) {
      const x = posAttr.getX(i)
      const y = posAttr.getY(i)
      const z = posAttr.getZ(i)
      const noise = 1 + (Math.random() - 0.5) * 0.45
      posAttr.setXYZ(i, x * noise, y * noise, z * noise)
    }
    asteroidGeo.computeVertexNormals()

    const asteroidMat = new THREE.MeshStandardMaterial({
      color: 0x7a6a55,
      roughness: 0.98,
      metalness: 0.02,
      flatShading: true,
    })
    const asteroid = new THREE.Mesh(asteroidGeo, asteroidMat)
    asteroid.position.set(1.5, 0.2, 0)
    scene.add(asteroid)

    // ── Glow atmosphere ───────────────────────────────────────────────────────
    const glowGeo = new THREE.SphereGeometry(2.9, 32, 32)
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0x3b82f6,
      transparent: true,
      opacity: 0.06,
      side: THREE.BackSide,
    })
    const glow = new THREE.Mesh(glowGeo, glowMat)
    glow.position.copy(asteroid.position)
    scene.add(glow)

    // ── Orbit ring ────────────────────────────────────────────────────────────
    const ringGeo = new THREE.TorusGeometry(3.6, 0.015, 8, 120)
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x3b82f6,
      transparent: true,
      opacity: 0.25,
    })
    const ring = new THREE.Mesh(ringGeo, ringMat)
    ring.rotation.x = Math.PI * 0.42
    ring.position.copy(asteroid.position)
    scene.add(ring)

    // ── Small debris asteroids ────────────────────────────────────────────────
    const debris: THREE.Mesh[] = []
    const debrisData = [
      { pos: [-4, 2, -3] as [number, number, number], scale: 0.25, speed: 0.6 },
      { pos: [4.5, -1.5, -2] as [number, number, number], scale: 0.18, speed: -0.8 },
      { pos: [-2.5, -3, -4] as [number, number, number], scale: 0.3, speed: 0.4 },
    ]
    for (const d of debrisData) {
      const geo = new THREE.IcosahedronGeometry(1, 1)
      const pos = geo.getAttribute("position") as THREE.BufferAttribute
      for (let i = 0; i < pos.count; i++) {
        const nx = pos.getX(i) * (1 + (Math.random() - 0.5) * 0.5)
        const ny = pos.getY(i) * (1 + (Math.random() - 0.5) * 0.5)
        const nz = pos.getZ(i) * (1 + (Math.random() - 0.5) * 0.5)
        pos.setXYZ(i, nx, ny, nz)
      }
      geo.computeVertexNormals()
      const mesh = new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({ color: 0x5a4e3f, roughness: 0.95, flatShading: true })
      )
      mesh.position.set(...d.pos)
      mesh.scale.setScalar(d.scale)
      mesh.userData = { speed: d.speed }
      scene.add(mesh)
      debris.push(mesh)
    }

    // ── Lights ────────────────────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0x223355, 1.2))
    const sun = new THREE.DirectionalLight(0xfff0cc, 4)
    sun.position.set(10, 8, 6)
    scene.add(sun)
    const fill = new THREE.DirectionalLight(0x4488ff, 1.5)
    fill.position.set(-8, -4, -6)
    scene.add(fill)

    // ── Animation ─────────────────────────────────────────────────────────────
    const clock = new THREE.Clock()
    let animId: number

    function animate() {
      animId = requestAnimationFrame(animate)
      const t = clock.getElapsedTime()

      asteroid.rotation.y = t * 0.12
      asteroid.rotation.x = t * 0.04
      asteroid.position.y = 0.2 + Math.sin(t * 0.5) * 0.25
      glow.position.copy(asteroid.position)
      ring.position.copy(asteroid.position)
      ring.rotation.z = t * 0.05

      for (const d of debris) {
        d.rotation.y += 0.005 * (d.userData.speed as number)
        d.rotation.x += 0.003 * (d.userData.speed as number)
      }

      stars.rotation.y = t * 0.008
      stars.rotation.x = t * 0.002

      renderer.render(scene, camera)
    }
    animate()

    // ── Resize ────────────────────────────────────────────────────────────────
    function onResize() {
      const nw = mount.clientWidth
      const nh = mount.clientHeight
      camera.aspect = nw / nh
      camera.updateProjectionMatrix()
      renderer.setSize(nw, nh)
    }
    window.addEventListener("resize", onResize)

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener("resize", onResize)
      renderer.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={mountRef} className="absolute inset-0" />
}
