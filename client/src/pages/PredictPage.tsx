import { useState } from "react"
import { AlertTriangle, CheckCircle2, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import Navbar from "@/components/Navbar"

interface FormState {
  est_diameter_min: string
  est_diameter_max: string
  relative_velocity: string
  absolute_magnitude: string
  miss_distance: string
}

interface PredictionResult {
  hazardous: boolean
  hazardous_probability: number
  miss_distance_km: number
}

const DEFAULT: FormState = {
  est_diameter_min: "0.12",
  est_diameter_max: "0.27",
  relative_velocity: "48000",
  absolute_magnitude: "22.1",
  miss_distance: "14500000",
}

const FIELDS: { name: keyof FormState; label: string; unit: string; hint: string }[] = [
  { name: "est_diameter_min", label: "Min Estimated Diameter", unit: "km", hint: "Minimum diameter estimate" },
  { name: "est_diameter_max", label: "Max Estimated Diameter", unit: "km", hint: "Maximum diameter estimate" },
  { name: "relative_velocity", label: "Relative Velocity", unit: "km/h", hint: "Speed relative to Earth" },
  { name: "absolute_magnitude", label: "Absolute Magnitude", unit: "H", hint: "Intrinsic brightness (lower = brighter/larger)" },
  { name: "miss_distance", label: "Miss Distance", unit: "km", hint: "Closest approach distance" },
]

export default function PredictPage() {
  const [form, setForm] = useState<FormState>(DEFAULT)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          est_diameter_min: parseFloat(form.est_diameter_min),
          est_diameter_max: parseFloat(form.est_diameter_max),
          relative_velocity: parseFloat(form.relative_velocity),
          absolute_magnitude: parseFloat(form.absolute_magnitude),
          miss_distance: parseFloat(form.miss_distance),
        }),
      })
      if (!res.ok) {
        const body = (await res.json()) as { detail?: string }
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      setResult((await res.json()) as PredictionResult)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error occurred")
    } finally {
      setLoading(false)
    }
  }

  const probPercent = result ? Math.round(result.hazardous_probability * 100) : 0

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />

      {/* Background grid */}
      <div
        className="fixed inset-0 opacity-[0.03] pointer-events-none"
        style={{
          backgroundImage: `linear-gradient(hsl(var(--border)) 1px, transparent 1px), linear-gradient(90deg, hsl(var(--border)) 1px, transparent 1px)`,
          backgroundSize: "48px 48px",
        }}
      />

      <main className="relative z-10 pt-24 pb-16 px-6">
        <div className="max-w-xl mx-auto">

          {/* Header */}
          <div className="text-center mb-8">
            <Badge variant="outline" className="mb-4 text-xs tracking-widest uppercase border-primary/40 text-primary">
              ML Inference
            </Badge>
            <h1 className="text-3xl font-bold mb-2">Asteroid Analysis</h1>
            <p className="text-muted-foreground text-sm">
              Enter raw observation data to get an instant hazard prediction
            </p>
          </div>

          {/* Form card */}
          <Card className="border-border/60 bg-card/70 backdrop-blur-sm shadow-xl shadow-black/20">
            <CardHeader>
              <CardTitle className="text-lg">Observation Data</CardTitle>
              <CardDescription>All five fields are required for inference</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  {FIELDS.slice(0, 2).map((f) => (
                    <div key={f.name} className="space-y-1.5">
                      <Label htmlFor={f.name}>
                        {f.label}{" "}
                        <span className="text-muted-foreground font-normal">({f.unit})</span>
                      </Label>
                      <Input
                        id={f.name}
                        name={f.name}
                        type="number"
                        step="any"
                        placeholder={f.hint}
                        value={form[f.name]}
                        onChange={handleChange}
                        required
                      />
                    </div>
                  ))}
                </div>

                {FIELDS.slice(2).map((f) => (
                  <div key={f.name} className="space-y-1.5">
                    <Label htmlFor={f.name}>
                      {f.label}{" "}
                      <span className="text-muted-foreground font-normal">({f.unit})</span>
                    </Label>
                    <Input
                      id={f.name}
                      name={f.name}
                      type="number"
                      step="any"
                      placeholder={f.hint}
                      value={form[f.name]}
                      onChange={handleChange}
                      required
                    />
                  </div>
                ))}

                <Button type="submit" disabled={loading} className="w-full mt-2 gap-2 shadow-lg shadow-primary/20">
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Running Inference…
                    </>
                  ) : (
                    "Predict Hazard"
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Error */}
          {error && (
            <Card className="mt-4 border-destructive/50 bg-destructive/10">
              <CardContent className="pt-4 pb-4 flex items-center gap-3 text-destructive">
                <AlertTriangle className="h-5 w-5 shrink-0" />
                <p className="text-sm">{error}</p>
              </CardContent>
            </Card>
          )}

          {/* Result */}
          {result && (
            <Card className={`mt-4 shadow-xl ${
              result.hazardous
                ? "border-red-500/50 bg-red-500/5 shadow-red-500/10"
                : "border-emerald-500/50 bg-emerald-500/5 shadow-emerald-500/10"
            }`}>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  {result.hazardous ? (
                    <AlertTriangle className="h-6 w-6 text-red-400 shrink-0" />
                  ) : (
                    <CheckCircle2 className="h-6 w-6 text-emerald-400 shrink-0" />
                  )}
                  <div>
                    <CardTitle className={result.hazardous ? "text-red-400" : "text-emerald-400"}>
                      {result.hazardous ? "Potentially Hazardous Object" : "Non-Hazardous Object"}
                    </CardTitle>
                    <CardDescription>
                      {result.hazardous
                        ? "This asteroid meets NASA's PHO criteria"
                        : "This asteroid poses no significant threat"}
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="space-y-5">
                {/* Probability */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Hazard Probability</span>
                    <Badge variant={result.hazardous ? "warning" : "success"}>
                      {probPercent}%
                    </Badge>
                  </div>
                  <Progress
                    value={probPercent}
                    className={result.hazardous ? "[&>div]:bg-red-500" : "[&>div]:bg-emerald-500"}
                  />
                </div>

                {/* Miss distance */}
                <div className="rounded-lg border border-border/50 bg-background/40 p-4">
                  <div className="text-xs text-muted-foreground uppercase tracking-widest mb-1">
                    Predicted Miss Distance
                  </div>
                  <div className="text-2xl font-bold">
                    {result.miss_distance_km >= 1_000_000
                      ? `${(result.miss_distance_km / 1_000_000).toFixed(2)}M km`
                      : `${result.miss_distance_km.toLocaleString()} km`}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {(result.miss_distance_km / 384400).toFixed(1)} lunar distances
                  </div>
                </div>

                {/* Raw probability bar detail */}
                <div className="text-xs text-muted-foreground text-center pt-1">
                  Model confidence: {(result.hazardous_probability * 100).toFixed(2)}% probability of hazard
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  )
}
