# syntax=docker/dockerfile:1
FROM node:20-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

RUN npm ci --only=production

# Copy source code
COPY . .

EXPOSE 5000

CMD ["node", "src/server.js"]
