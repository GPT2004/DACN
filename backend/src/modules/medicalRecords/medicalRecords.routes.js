const express = require('express');
const router = express.Router();
const service = require('./medicalRecords.service');

// GET /api/medical-records?patientId=&q=&limit=&offset=
router.get('/', async (req, res) => {
  try {
    const { patientId, q, limit, offset } = req.query;
    const data = await service.listMedicalRecords({ patientId, q, limit: Number(limit) || 50, offset: Number(offset) || 0 });
    res.json({ success: true, data });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

router.get('/:id', async (req, res) => {
  try {
    const rec = await service.getMedicalRecord(req.params.id);
    if (!rec) return res.status(404).json({ success: false, message: 'Not found' });
    res.json({ success: true, data: rec });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

router.post('/', async (req, res) => {
  try {
    const payload = req.body;
    const created = await service.createMedicalRecord(payload);
    res.status(201).json({ success: true, data: created });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

router.put('/:id', async (req, res) => {
  try {
    const updated = await service.updateMedicalRecord(req.params.id, req.body);
    if (!updated) return res.status(404).json({ success: false, message: 'Not found' });
    res.json({ success: true, data: updated });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

router.delete('/:id', async (req, res) => {
  try {
    await service.deleteMedicalRecord(req.params.id);
    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

module.exports = router;
