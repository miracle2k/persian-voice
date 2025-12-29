# Stage 1: Build the static site
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy source files
COPY . .

# Build static export
ARG NEXT_PUBLIC_AUDIO_BASE_URL
ARG NEXT_PUBLIC_TURSO_DATABASE_URL
ARG NEXT_PUBLIC_TURSO_AUTH_TOKEN

ENV NEXT_PUBLIC_AUDIO_BASE_URL=$NEXT_PUBLIC_AUDIO_BASE_URL
ENV NEXT_PUBLIC_TURSO_DATABASE_URL=$NEXT_PUBLIC_TURSO_DATABASE_URL
ENV NEXT_PUBLIC_TURSO_AUTH_TOKEN=$NEXT_PUBLIC_TURSO_AUTH_TOKEN

RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static files from builder
COPY --from=builder /app/out /usr/share/nginx/html

# Expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
