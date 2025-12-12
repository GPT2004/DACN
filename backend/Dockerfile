FROM node:18-alpine3.18

# Install OpenSSL 1.1 compatibility for Prisma engine binaries (available on Alpine 3.18)
# Also install Python for AI predictions

WORKDIR /app

COPY package*.json ./

RUN npm ci --only=production

COPY . .

RUN npx prisma generate

EXPOSE 3000

CMD ["npm", "start"]