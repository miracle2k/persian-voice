import { getTursoClient } from '@/lib/turso';
import { NextResponse } from 'next/server';

export async function POST(request) {
  const client = getTursoClient();
  if (!client) {
    return NextResponse.json({ error: 'Database not configured' }, { status: 503 });
  }

  try {
    const { cellId, hasIssue } = await request.json();

    if (!cellId) {
      return NextResponse.json({ error: 'cellId required' }, { status: 400 });
    }

    const [wordId, modelId] = cellId.split('||');
    if (!wordId || !modelId) {
      return NextResponse.json({ error: 'Invalid cellId format' }, { status: 400 });
    }

    if (hasIssue) {
      await client.execute({
        sql: `INSERT INTO cell_issues (id, word_id, model_id, updated_at)
              VALUES (?, ?, ?, datetime('now'))
              ON CONFLICT(id) DO NOTHING`,
        args: [cellId, wordId, modelId]
      });
    } else {
      await client.execute({
        sql: 'DELETE FROM cell_issues WHERE id = ?',
        args: [cellId]
      });
    }

    return NextResponse.json({ success: true });
  } catch (err) {
    console.error('Failed to save cell issue:', err);
    return NextResponse.json({ error: 'Database error' }, { status: 500 });
  }
}
