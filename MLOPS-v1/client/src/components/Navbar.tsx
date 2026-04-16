import { Link, useLocation } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Telescope } from "lucide-react"

export default function Navbar() {
  const location = useLocation()
  const isPredict = location.pathname === "/predict"

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 bg-background/60 backdrop-blur-md border-b border-border/40">
      <Link to="/" className="flex items-center gap-2 text-foreground hover:text-primary transition-colors">
        <Telescope className="h-5 w-5 text-primary" />
        <span className="font-semibold text-sm tracking-wide">NEO Predictor</span>
      </Link>

      <div className="flex items-center gap-4">
        <a
          href="/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          API Docs
        </a>
        {isPredict ? (
          <Link to="/">
            <Button variant="ghost" size="sm">← Home</Button>
          </Link>
        ) : (
          <Link to="/predict">
            <Button size="sm">Analyze Asteroid</Button>
          </Link>
        )}
      </div>
    </nav>
  )
}
