# Stage 1: Build the frontend
FROM node:20-slim AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: Production environment
FROM node:20-slim

WORKDIR /app

# Install only production dependencies
COPY package*.json ./
RUN npm install --production

# Copy built assets from stage 1
COPY --from=build /app/dist ./dist
COPY --from=build /app/public ./public
COPY --from=build /app/src ./src

# Create models directory for uploads
RUN mkdir -p public/models

EXPOSE 3000

# The dynamic config.js is served from src/server.js
# But we can allow overriding it by mounting a file to /app/public/config.js
VOLUME ["/app/public/config.js", "/app/public/models"]

CMD ["node", "src/server.js"]
