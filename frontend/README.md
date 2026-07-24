# ChurnAware Frontend

Next.js dashboard for the ChurnAware retention platform, built with shadcn/ui, Tailwind CSS and Recharts. Light and dark themes are both supported.

## Development

Start the API first from the repository root:

```bash
uvicorn backend.app.main:app --port 8000
```

Then run the frontend:

```bash
cd frontend
npm install
npm run dev
```

The app expects the API at `http://127.0.0.1:8000` by default; override with the `NEXT_PUBLIC_API_URL` environment variable.

## Pages

- `/` — portfolio KPIs and distribution charts
- `/segments` — GMM segment profiles and comparisons
- `/risk` — churn model quality, feature importance, at-risk queue
- `/retention` — RL policy evaluation and live action recommendations
- `/customers` — filterable, paginated customer explorer
