import { Link } from "react-router-dom"
import { ArrowRight, Brain, Ruler, Zap, ChevronDown, Shield, Database } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import Navbar from "@/components/Navbar"
import AsteroidScene from "@/components/AsteroidScene"

const stats = [
  { value: "90,836", label: "Asteroids Analyzed" },
  { value: "2", label: "ML Models" },
  { value: "<50ms", label: "Prediction Latency" },
  { value: "99%+", label: "Classification F1" },
]

const features = [
  {
    icon: Shield,
    title: "Hazard Classification",
    description:
      "Gradient Boosting classifier predicts whether an asteroid is a Potentially Hazardous Object with >99% F1-score, trained on NASA's full NEO dataset.",
    badge: "Classification",
  },
  {
    icon: Ruler,
    title: "Distance Regression",
    description:
      "A separate regressor predicts the asteroid's closest approach distance to Earth in km, using log-space training for high dynamic-range accuracy.",
    badge: "Regression",
  },
  {
    icon: Zap,
    title: "Real-time API",
    description:
      "FastAPI backend with models loaded at startup. Redis-cached responses, PostgreSQL prediction logging, and a live WebSocket stream.",
    badge: "Live",
  },
]

const pipeline = [
  {
    step: "01",
    title: "Enter Observations",
    description: "Provide five raw asteroid measurements: diameter range, relative velocity, absolute magnitude, and miss distance.",
  },
  {
    step: "02",
    title: "ML Inference",
    description: "Features are engineered (log-transforms, derived ratios), scaled, and fed to two independently tuned Gradient Boosting models.",
  },
  {
    step: "03",
    title: "Instant Results",
    description: "Receive hazard classification with probability score and predicted closest-approach distance — all in under 50ms.",
  },
]

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="relative h-screen flex items-center overflow-hidden">
        <AsteroidScene />

        {/* Gradient overlay — darkens the left side so text is readable */}
        <div className="absolute inset-0 bg-gradient-to-r from-background via-background/70 to-transparent z-10 pointer-events-none" />

        {/* Hero content */}
        <div className="relative z-20 max-w-7xl mx-auto px-6 w-full">
          <div className="max-w-xl">
            <Badge variant="outline" className="mb-6 text-xs tracking-widest uppercase border-primary/40 text-primary animate-fade-in">
              <Brain className="h-3 w-3 mr-1" />
              NASA Near-Earth Object Data
            </Badge>

            <h1 className="text-5xl md:text-7xl font-bold leading-[1.05] tracking-tight mb-6 animate-fade-up">
              Will This
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
                Asteroid
              </span>
              <br />
              Hit Earth?
            </h1>

            <p className="text-lg text-muted-foreground mb-8 leading-relaxed animate-fade-up-delay-1">
              Machine learning–powered hazard prediction and closest-approach
              distance estimation for near-Earth objects.
            </p>

            <div className="flex flex-wrap gap-3 animate-fade-up-delay-2">
              <Link to="/predict">
                <Button size="lg" className="gap-2 shadow-lg shadow-primary/25">
                  Analyze an Asteroid
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <a href="/docs" target="_blank" rel="noopener noreferrer">
                <Button size="lg" variant="outline" className="gap-2">
                  <Database className="h-4 w-4" />
                  API Docs
                </Button>
              </a>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-1 text-muted-foreground animate-float">
          <span className="text-xs tracking-widest uppercase">Scroll</span>
          <ChevronDown className="h-4 w-4" />
        </div>
      </section>

      {/* ── Stats bar ────────────────────────────────────────────────────── */}
      <section className="border-y border-border/50 bg-card/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((s) => (
            <div key={s.label} className="text-center">
              <div className="text-3xl font-bold text-primary mb-1">{s.value}</div>
              <div className="text-sm text-muted-foreground">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ─────────────────────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-24">
        <div className="text-center mb-14">
          <Badge variant="outline" className="mb-4 text-xs tracking-widest uppercase border-primary/40 text-primary">
            Capabilities
          </Badge>
          <h2 className="text-4xl font-bold mb-4">Two Models. One API.</h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Independently optimised models for classification and regression,
            served through a single prediction endpoint.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {features.map((f) => (
            <Card key={f.title} className="group border-border/50 hover:border-primary/40 transition-all duration-300 hover:shadow-lg hover:shadow-primary/5 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <div className="h-10 w-10 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <f.icon className="h-5 w-5 text-primary" />
                </div>
                <div className="flex items-center gap-2 mb-1">
                  <CardTitle>{f.title}</CardTitle>
                </div>
                <Badge variant="secondary" className="w-fit text-xs">{f.badge}</Badge>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-sm leading-relaxed">{f.description}</CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* ── How it works ─────────────────────────────────────────────────── */}
      <section className="border-t border-border/50 bg-card/30">
        <div className="max-w-7xl mx-auto px-6 py-24">
          <div className="text-center mb-14">
            <Badge variant="outline" className="mb-4 text-xs tracking-widest uppercase border-primary/40 text-primary">
              Workflow
            </Badge>
            <h2 className="text-4xl font-bold mb-4">How It Works</h2>
          </div>

          <div className="grid md:grid-cols-3 gap-8 relative">
            {/* connector line */}
            <div className="hidden md:block absolute top-8 left-1/4 right-1/4 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent" />

            {pipeline.map((p) => (
              <div key={p.step} className="flex flex-col items-center text-center">
                <div className="h-16 w-16 rounded-full border-2 border-primary/40 bg-primary/10 flex items-center justify-center mb-5 text-2xl font-bold text-primary">
                  {p.step}
                </div>
                <h3 className="font-semibold text-lg mb-2">{p.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{p.description}</p>
              </div>
            ))}
          </div>

          <div className="flex justify-center mt-12">
            <Link to="/predict">
              <Button size="lg" className="gap-2 shadow-lg shadow-primary/25">
                Try the Predictor
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="border-t border-border/50 bg-background/80">
        <div className="max-w-7xl mx-auto px-6 py-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Brain className="h-4 w-4 text-primary" />
            <span>NEO Predictor — Gradient Boosting + FastAPI + React</span>
          </div>
          <div className="flex gap-6 text-sm text-muted-foreground">
            <a href="/docs" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">API</a>
            <Link to="/predict" className="hover:text-foreground transition-colors">Predictor</Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
