# Stage 1: Build the frontend
FROM registry.cern.ch/docker.io/node:20-slim AS build

WORKDIR /app

ARG NPM_AUTH_TOKEN
ENV NPM_AUTH_TOKEN=${NPM_AUTH_TOKEN}

COPY package*.json ./

COPY . .

#COPY assets ./
#COPY public ./
#COPY src ./
#COPY webpack.config.js ./
#COPY .npmrc ./

RUN npm install
RUN npm run build

# Stage 2: Production environment
FROM registry.cern.ch/docker.io/node:20-slim

ARG NPM_AUTH_TOKEN
ENV NPM_AUTH_TOKEN=${NPM_AUTH_TOKEN}

WORKDIR /app

# Install dependencies
COPY package*.json ./
COPY .npmrc ./
RUN npm install

# Copy built assets from stage 1
COPY --from=build /app/dist ./dist
COPY --from=build /app/public ./public
COPY --from=build /app/src ./src
COPY --from=build /app/assets ./assets
COPY --from=build /app/webpack.config.js ./webpack.config.js

# Create models directory for uploads
RUN mkdir -p public/models

EXPOSE 3000
EXPOSE 9000

# The dynamic config.js is served from src/server.js
# But we can allow overriding it by mounting a file to /app/public/config.js
VOLUME ["/app/public/config.js", "/app/public/models"]

ENTRYPOINT ["npm", "run", "start"]
