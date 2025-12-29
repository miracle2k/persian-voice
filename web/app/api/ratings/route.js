import { getTursoClient } from '@/lib/turso';
import { NextResponse } from 'next/server';

export async function GET() {
  const client = getTursoClient();
  if (!client) {
    return NextResponse.json({ modelRatings: {}, cellIssues: [] });
  }

  try {
    const [ratingsResult, issuesResult] = await Promise.all([
      client.execute('SELECT model_id, rating FROM model_ratings'),
      client.execute('SELECT id FROM cell_issues')
    ]);

    const modelRatings = {};
    for (const row of ratingsResult.rows) {
      modelRatings[row.model_id] = row.rating;
    }

    const cellIssues = issuesResult.rows.map(row => row.id);

    return NextResponse.json({ modelRatings, cellIssues });
  } catch (err) {
    console.error('Failed to fetch ratings:', err);
    return NextResponse.json({ modelRatings: {}, cellIssues: [] }, { status: 500 });
  }
}
