import { createClient } from '@libsql/client';

let client = null;

export function getTursoClient() {
  if (!client && process.env.TURSO_DATABASE_URL) {
    client = createClient({
      url: process.env.TURSO_DATABASE_URL,
      authToken: process.env.TURSO_AUTH_TOKEN,
    });
  }
  return client;
}
