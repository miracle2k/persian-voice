import "./globals.css";

export const metadata = {
  title: "Persian Voice Comparison",
  description: "Compare Persian TTS providers/models on the same word list."
};

export default function RootLayout({ children }) {
  return (
    <html lang="fa">
      <body>{children}</body>
    </html>
  );
}

