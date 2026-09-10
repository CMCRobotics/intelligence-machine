# Stage 1: Build the frontend and install dependencies
FROM oven/bun:slim AS build

WORKDIR /app

# Accept the NPM Auth Token as a build argument
ARG NPM_AUTH_TOKEN
ENV NPM_AUTH_TOKEN=${NPM_AUTH_TOKEN}

# Copy package configuration and npmrc auth settings
COPY package.json ./
COPY .npmrc ./

# Install all dependencies (including packages on the private GitHub package registry)
RUN bun install

# Copy application source and assets
COPY . .

# Build the frontend bundles
RUN bun run build

# Stage 2: Production environment
FROM oven/bun:slim

WORKDIR /app

# Copy built assets and runtime files from Stage 1
COPY --from=build /app/dist ./dist
COPY --from=build /app/public ./public
COPY --from=build /app/src ./src
COPY --from=build /app/assets ./assets
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/node_modules ./node_modules

# Ensure model directory exists
RUN mkdir -p public/models

EXPOSE 3000
EXPOSE 9000

# Mount volumes for persistent assets
VOLUME ["/app/public/config.js", "/app/public/models"]

ENTRYPOINT ["bun", "run", "start"]


