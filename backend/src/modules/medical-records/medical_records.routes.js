const express = require('express');
const router = express.Router();
const service = require('./medical_records.service');
const { authenticate, optionalAuth } = require('../../middlewares/auth.middleware');
const { validate } = require('../../middlewares/validate.middleware');
const { createMedicalRecordSchema, updateMedicalRecordSchema } = require('./medical_records.validator');

// GET /api/medical-records
router.get('/', optionalAuth, async (req, res) => {
  try {
    const { patientId, doctorId, q, limit = 10, offset = 0, page } = req.query;
    const pageNum = page ? Number(page) : Math.floor(Number(offset) / Number(limit || 10)) + 1;
    const filters = {
      patient_id: patientId ? Number(patientId) : undefined,
      doctor_id: doctorId ? Number(doctorId) : undefined,
      q,
      page: pageNum,
      limit: Number(limit) || 10,
    };
    // pass req.user so service can apply role-based filtering
    const data = await service.getMedicalRecords(filters, req.user);
    res.json({ success: true, data });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

router.get('/:id', optionalAuth, async (req, res) => {
  try {
    const rec = await service.getMedicalRecordById(req.params.id, req.user);
    if (!rec) return res.status(404).json({ success: false, message: 'Not found' });
    res.json({ success: true, data: rec });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: err.message || 'Server error' });
  }
});

// Create medical record (authenticated doctors only)
router.post('/', authenticate, validate(createMedicalRecordSchema), async (req, res) => {
  try {
    const payload = req.validatedBody;
    const created = await service.createMedicalRecord(payload, req.user);
    res.status(201).json({ success: true, data: created });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: err.message || 'Server error' });
  }
});

router.put('/:id', authenticate, validate(updateMedicalRecordSchema), async (req, res) => {
  try {
    const updated = await service.updateMedicalRecord(req.params.id, req.validatedBody, req.user);
    if (!updated) return res.status(404).json({ success: false, message: 'Not found' });
    res.json({ success: true, data: updated });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: err.message || 'Server error' });
  }
});

router.delete('/:id', authenticate, async (req, res) => {
  try {
    await service.deleteMedicalRecord(req.params.id, req.user);
    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: err.message || 'Server error' });
  }
});

router.post('/:id/send-to-patient', authenticate, async (req, res) => {
  try {
    const result = await service.sendMedicalRecordToPatient(req.params.id, req.user);
    res.json({ success: true, data: result });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: err.message || 'Server error' });
  }
});

module.exports = router;