import { getTursoClient } from '@/lib/turso';
import { NextResponse } from 'next/server';

export async function POST(request) {
  const client = getTursoClient();
  if (!client) {
    return NextResponse.json({ error: 'Database not configured' }, { status: 503 });
  }

  try {
    const { modelId, rating } = await request.json();

    if (!modelId) {
      return NextResponse.json({ error: 'modelId required' }, { status: 400 });
    }

    if (rating === null) {
      await client.execute({
        sql: 'DELETE FROM model_ratings WHERE model_id = ?',
        args: [modelId]
      });
    } else if (['good', 'mediocre', 'broken'].includes(rating)) {
      await client.execute({
        sql: `INSERT INTO model_ratings (model_id, rating, updated_at)
              VALUES (?, ?, datetime('now'))
              ON CONFLICT(model_id) DO UPDATE SET rating = excluded.rating, updated_at = datetime('now')`,
        args: [modelId, rating]
      });
    } else {
      return NextResponse.json({ error: 'Invalid rating' }, { status: 400 });
    }

    return NextResponse.json({ success: true });
  } catch (err) {
    console.error('Failed to save model rating:', err);
    return NextResponse.json({ error: 'Database error' }, { status: 500 });
  }
}
