/*
  Warnings:

  - You are about to drop the column `new_district` on the `patients` table. All the data in the column will be lost.
  - You are about to drop the column `new_province` on the `patients` table. All the data in the column will be lost.
  - You are about to drop the column `new_street` on the `patients` table. All the data in the column will be lost.
  - You are about to drop the column `new_ward` on the `patients` table. All the data in the column will be lost.
  - You are about to drop the column `old_district` on the `patients` table. All the data in the column will be lost.
  - You are about to drop the column `old_province` on the `patients` table. All the data in the column will be lost.
  - You are about to drop the column `old_street` on the `patients` table. All the data in the column will be lost.
  - You are about to drop the column `old_ward` on the `patients` table. All the data in the column will be lost.
  - A unique constraint covering the columns `[user_id]` on the table `patients` will be added. If there are existing duplicate values, this will fail.

*/
-- AlterEnum
-- This migration adds more than one value to an enum.
-- With PostgreSQL versions 11 and earlier, this is not possible
-- in a single migration. This can be worked around by creating
-- multiple migrations, each migration adding only one value to
-- the enum.


ALTER TYPE "prescription_status" ADD VALUE 'INVOICED';
ALTER TYPE "prescription_status" ADD VALUE 'READY_FOR_DISPENSE';

-- DropForeignKey
ALTER TABLE "patients" DROP CONSTRAINT "fk_patients_owner_user_id";

-- DropIndex
DROP INDEX "idx_patients_id_number";

-- DropIndex
DROP INDEX "idx_patients_new_province";

-- DropIndex
DROP INDEX "idx_patients_old_province";

-- DropIndex
DROP INDEX "idx_patients_owner_user_id";

-- AlterTable
ALTER TABLE "medicines" ADD COLUMN     "deleted_at" TIMESTAMPTZ(6);

-- AlterTable
ALTER TABLE "patients" DROP COLUMN "new_district",
DROP COLUMN "new_province",
DROP COLUMN "new_street",
DROP COLUMN "new_ward",
DROP COLUMN "old_district",
DROP COLUMN "old_province",
DROP COLUMN "old_street",
DROP COLUMN "old_ward";

-- AlterTable
ALTER TABLE "prescriptions" ADD COLUMN     "notified_at" TIMESTAMPTZ(6),
ADD COLUMN     "notified_by" INTEGER;

-- AlterTable
ALTER TABLE "rooms" ADD COLUMN     "deleted_at" TIMESTAMPTZ(6),
ADD COLUMN     "updated_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP;

-- AlterTable
ALTER TABLE "specialties" ADD COLUMN     "deleted_at" TIMESTAMPTZ(6);

-- AlterTable
ALTER TABLE "stocks" ADD COLUMN     "deleted_at" TIMESTAMPTZ(6);

-- AlterTable
ALTER TABLE "users" ADD COLUMN     "deleted_at" TIMESTAMPTZ(6);

-- CreateIndex
CREATE UNIQUE INDEX "patients_user_id_key" ON "patients"("user_id");

-- AddForeignKey
ALTER TABLE "patients" ADD CONSTRAINT "patients_owner_user_id_fkey" FOREIGN KEY ("owner_user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE NO ACTION;

-- AddForeignKey
ALTER TABLE "prescriptions" ADD CONSTRAINT "prescriptions_notified_by_fkey" FOREIGN KEY ("notified_by") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE NO ACTION;

-- RenameIndex
ALTER INDEX "uq_doctor_patient_review" RENAME TO "doctor_reviews_doctor_id_patient_id_key";
