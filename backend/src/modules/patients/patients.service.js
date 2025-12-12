/* eslint-disable no-console */
const prisma = require('../../config/database');
const { hashPassword } = require('../../utils/bcrypt');
const { ROLES } = require('../../config/constants');

class PatientsService {
  // Helper: return array of existing column names from patients table for the given list
  async _getExistingPatientColumns(txOrClient, cols) {
    if (!cols || cols.length === 0) return [];
    const colsCondition = cols.map(c => `'${c}'`).join(',');
    const client = txOrClient || prisma;
    const rows = await client.$queryRawUnsafe(`SELECT column_name FROM information_schema.columns WHERE table_name = 'patients' AND column_name IN (${colsCondition})`);
    return Array.isArray(rows) ? rows.map(r => r.column_name) : [];
  }

  async getAllPatients(filters = {}, pagination = { page: 1, limit: 10 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {};

      if (filters.gender) {
        where.gender = filters.gender;
      }

      if (filters.blood_type) {
        where.blood_type = filters.blood_type;
      }

      if (filters.search) {
        where.user = {  // ← Fixed: users → user
          OR: [
            {
              full_name: {
                contains: filters.search,
                mode: 'insensitive'
              }
            },
            {
              phone: {
                contains: filters.search,
                mode: 'insensitive'
              }
            },
            {
              email: {
                contains: filters.search,
                mode: 'insensitive'
              }
            }
          ]
        };
      }

      const [total, patients] = await Promise.all([
        prisma.patients.count({ where }),
        prisma.patients.findMany({
          where,
          skip,
          take: limit,
          include: {
            user: {  // ← Fixed: users → user
              select: {
                id: true,
                full_name: true,
                email: true,
                phone: true,
                dob: true,
                avatar_url: true,
                is_active: true,
                created_at: true
              }
            },
            _count: {
              select: {
                appointments: true,
                medical_records: true
              }
            }
          },
          orderBy: {
            created_at: 'desc'
          }
        })
      ]);

      return {
        patients,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getAllPatients:', error);
      throw error;
    }
  }

  async getPatientById(id) {
    try {
      const patient = await prisma.patients.findUnique({
        where: { id },
        include: {
          user: {  // ← Fixed: users → user
            select: {
              id: true,
              full_name: true,
              email: true,
              phone: true,
              dob: true,
              avatar_url: true,
              is_active: true,
              created_at: true
            }
          },
          _count: {
            select: {
              appointments: true,
              medical_records: true,
              prescriptions: true,
              invoices: true
            }
          }
        }
      });

      return patient;
    } catch (error) {
      console.error('Error in getPatientById:', error);
      throw error;
    }
  }

  async getPatientByUserId(userId) {
    try {
      const patient = await prisma.patients.findFirst({
        where: { user_id: userId },
        include: {
          user: {  // ← Fixed: users → user
            select: {
              id: true,
              full_name: true,
              email: true,
              phone: true,
              dob: true,
              avatar_url: true
            }
          },
          _count: {
            select: {
              appointments: true,
              medical_records: true
            }
          }
        }
      });

      return patient;
    } catch (error) {
      console.error('Error in getPatientByUserId:', error);
      throw error;
    }
  }

  // Ensure a patient row exists for a given user_id. If missing, create a minimal patient record
  // populated from the users table (full_name, phone, email, dob) and return it.
  async ensurePatientForUser(userId) {
    try {
      let patient = await prisma.patients.findFirst({ where: { user_id: userId } });
      if (patient) {
        return await prisma.patients.findUnique({
          where: { id: patient.id },
          include: {
            user: {
              select: { id: true, full_name: true, email: true, phone: true, dob: true, avatar_url: true }
            }
          }
        });
      }

      // Fetch user info to populate patient row
      const user = await prisma.users.findUnique({ where: { id: userId } });
      if (!user) return null;

      const created = await prisma.patients.create({
        data: {
          user_id: userId,
          full_name: user.full_name || null,
          phone: user.phone || null,
          email: user.email || null,
          dob: user.dob || null,
          gender: null,
          blood_type: null,
          allergies: null,
          address: null,
          emergency_contact: {},
        },
        include: {
          user: {
            select: { id: true, full_name: true, email: true, phone: true, dob: true, avatar_url: true }
          }
        }
      });

      return created;
    } catch (error) {
      console.error('Error in ensurePatientForUser:', error);
      throw error;
    }
  }

  async getPatientMedicalRecords(patientId, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      // Include records owned by this patient row OR records shared to this patient via shared_medical_records
      let sharedIds = [];
      try {
        const rows = await prisma.$queryRaw`
          SELECT medical_record_id FROM shared_medical_records WHERE recipient_patient_id = ${patientId}
        `;
        if (Array.isArray(rows) && rows.length > 0) {
          sharedIds = rows.map((r) => r.medical_record_id).filter(Boolean);
        }
      } catch (err) {
        console.warn('Failed to fetch shared medical record ids:', err.message || err);
      }

      const where = sharedIds.length > 0
        ? { OR: [{ patient_id: patientId }, { id: { in: sharedIds } }] }
        : { patient_id: patientId };

      const [total, records] = await Promise.all([
        prisma.medical_records.count({ where }),
        prisma.medical_records.findMany({
          where,
          skip,
          take: limit,
          include: {
            doctor: {
              include: {
                user: {
                  select: {
                    full_name: true
                  }
                }
              }
            },
            appointment: {
              select: {
                appointment_date: true,
                appointment_time: true
              }
            }
          },
          orderBy: {
            created_at: 'desc'
          }
        })
      ]);

      // Fetch shared metadata for records that were shared to this patient
      let sharedMeta = {};
      if (sharedIds.length > 0) {
        try {
          const srows = await prisma.$queryRaw`
            SELECT medical_record_id, shared_by_user_id, created_at
            FROM shared_medical_records
            WHERE recipient_patient_id = ${patientId} AND medical_record_id = ANY(${sharedIds})
          `;
          if (Array.isArray(srows)) {
            for (const r of srows) {
              if (r && r.medical_record_id) {
                sharedMeta[r.medical_record_id] = {
                  shared_by_user_id: r.shared_by_user_id || null,
                  shared_at: r.created_at || null,
                };
              }
            }
          }
        } catch (err) {
          console.warn('Failed to fetch shared_medical_records metadata:', err.message || err);
        }
      }

      // Annotate records with shared info for frontend to render badge or details
      const annotated = (records || []).map((rec) => {
        const copy = Object.assign({}, rec);
        if (sharedIds.includes(rec.id)) {
          copy.shared = true;
          const meta = sharedMeta[rec.id] || {};
          copy.shared_by_user_id = meta.shared_by_user_id || null;
          copy.shared_at = meta.shared_at || null;
        } else {
          copy.shared = false;
        }
        return copy;
      });

      return {
        records: annotated,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getPatientMedicalRecords:', error);
      throw error;
    }
  }

  async getPatientAppointments(patientId, filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = { patient_id: patientId };

      if (filters.status) {
        where.status = filters.status;
      }

      if (filters.startDate || filters.endDate) {
        where.appointment_date = {};
        if (filters.startDate) where.appointment_date.gte = new Date(filters.startDate);
        if (filters.endDate) {
          // Adjust endDate to include the entire day (add 23:59:59.999)
          const endDate = new Date(filters.endDate);
          endDate.setHours(23, 59, 59, 999);
          where.appointment_date.lte = endDate;
        }
      }

      const [total, appointments] = await Promise.all([
        prisma.appointments.count({ where }),
        prisma.appointments.findMany({
          where,
          skip,
          take: limit,
          include: {
            doctor: {  // ← Fixed: doctors → doctor
              include: {
                user: {  // ← Fixed: users → user
                  select: {
                    full_name: true,
                    phone: true
                  }
                }
              }
            },
            timeslot: true  // ← Fixed: timeslots → timeslot
          },
          orderBy: [
            { appointment_date: 'desc' },
            { appointment_time: 'desc' }
          ]
        })
      ]);

      return {
        appointments,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getPatientAppointments:', error);
      throw error;
    }
  }

  async getPatientPrescriptions(patientId, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = { patient_id: patientId };

      const [total, prescriptions] = await Promise.all([
        prisma.prescriptions.count({ where }),
        prisma.prescriptions.findMany({
          where,
          skip,
          take: limit,
          include: {
            doctor: {  // ← Fixed: doctors → doctor
              include: {
                user: {  // ← Fixed: users → user
                  select: {
                    full_name: true
                  }
                }
              }
            },
            appointment: {  // ← Fixed: appointments → appointment
              select: {
                appointment_date: true
              }
            }
          },
          orderBy: {
            created_at: 'desc'
          }
        })
      ]);

      return {
        prescriptions,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getPatientPrescriptions:', error);
      throw error;
    }
  }

  async getPatientInvoices(patientId, filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = { patient_id: patientId };

      if (filters.status) {
        where.status = filters.status;
      }

      const [total, invoices] = await Promise.all([
        prisma.invoices.count({ where }),
        prisma.invoices.findMany({
          where,
          skip,
          take: limit,
          include: {
            appointment: {  // ← Fixed: appointments → appointment
              include: {
                doctor: {  // ← Fixed: doctors → doctor
                  include: {
                    user: {  // ← Fixed: users → user
                      select: {
                        full_name: true
                      }
                    }
                  }
                }
              }
            }
          },
          orderBy: {
            created_at: 'desc'
          }
        })
      ]);

      return {
        invoices,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getPatientInvoices:', error);
      throw error;
    }
  }

  async createPatient(patientData) {
    try {
      const {
        email,
        password,
        full_name,
        phone,
        dob,
        gender,
        blood_type,
        allergies,
        emergency_contact
      } = patientData;

      const existingUser = await prisma.users.findUnique({
        where: { email }
      });

      if (existingUser) {
        throw new Error('Email already registered');
      }

      const patientRole = await prisma.roles.findUnique({
        where: { name: ROLES.PATIENT }
      });

      if (!patientRole) {
        throw new Error('Patient role not found');
      }

      const hashedPassword = await hashPassword(password);

      // determine which extended columns actually exist in the patients table
      const extendedColumns = [
        'occupation','id_type','id_number','id_issue_date','id_issue_place','nationality','ethnicity','zalo'
      ];

      const colsCondition = extendedColumns.map(c => `'${c}'`).join(',');
      // use $queryRawUnsafe with static column names (safe here because names are constants)
      const existingRows = await prisma.$queryRawUnsafe(`SELECT column_name FROM information_schema.columns WHERE table_name = 'patients' AND column_name IN (${colsCondition})`);
      const existingCols = Array.isArray(existingRows) ? existingRows.map(r => r.column_name) : [];

      const result = await prisma.$transaction(async (tx) => {
        const user = await tx.users.create({
          data: {
            email,
            password: hashedPassword,
            full_name,
            phone,
            dob: dob ? new Date(dob) : null,
            role_id: patientRole.id,
            is_active: true
          }
        });

        const patient = await tx.patients.create({
          data: {
            user_id: user.id,
            gender,
            blood_type,
            allergies,
            emergency_contact: emergency_contact || {}
          },
          include: {
            user: {  // ← Fixed: users → user
              select: {
                id: true,
                full_name: true,
                email: true,
                phone: true
              }
            }
          }
        });

        return patient;
      });

      return result;
    } catch (error) {
      console.error('Error in createPatient:', error);
      throw error;
    }
  }

  async createDependentPatient(ownerUserId, patientData) {
    try {
      const {
        full_name,
        phone,
        dob,
        gender,
        blood_type,
        allergies,
        email,
        address,
        emergency_contact,
        occupation,
        id_type,
        id_number,
        id_issue_date,
        id_issue_place,
        nationality,
        ethnicity,
        zalo
      } = patientData;

      // Use a raw INSERT to avoid Prisma relation-input validation errors that
      // are triggered by the generated client when multiple user relations exist.
      // This performs a direct insert and returns the created row.
      // Use a conservative INSERT that only references columns known to exist
      // in older schemas to avoid failing when the optional columns were not
      // yet added to the database. Apply extended fields via profile update
      // once migrations are applied.
      const inserted = await prisma.$queryRaw`
        INSERT INTO "patients"
          ("owner_user_id", "user_id", "full_name", "phone", "dob", "gender", "blood_type", "allergies", "email", "address", "emergency_contact", "created_at", "updated_at")
        VALUES (
          ${ownerUserId},
          NULL,
          ${full_name},
          ${phone},
          ${dob ? new Date(dob) : null},
          ${gender},
          ${blood_type},
          ${allergies},
          ${email},
          ${address},
          ${JSON.stringify(emergency_contact || {})}::json,
          now(),
          now()
        )
        RETURNING *;
      `;

      // prisma.$queryRaw returns an array-like result; take the first row
      const created = Array.isArray(inserted) ? inserted[0] : inserted;

      // After creating the dependent row with conservative columns, try to apply extended fields
      try {
        const extendedUpdate = {};
        const extendedKeys = ['occupation','id_type','id_number','id_issue_date','id_issue_place','nationality','ethnicity','zalo','address'];
        for (const k of extendedKeys) {
          if (patientData[k] !== undefined && patientData[k] !== null) {
            if (k === 'id_issue_date') {
              const d = patientData[k] ? new Date(patientData[k]) : null;
              if (!isNaN(d)) extendedUpdate[k] = d;
              else if (d === null) extendedUpdate[k] = null;
            } else if (k === 'emergency_contact') {
              // handled below
            } else {
              extendedUpdate[k] = patientData[k];
            }
          }
        }

        if (patientData.emergency_contact) {
          extendedUpdate.emergency_contact = typeof patientData.emergency_contact === 'string'
            ? (patientData.emergency_contact ? JSON.parse(patientData.emergency_contact) : {})
            : patientData.emergency_contact;
        }

        const toCheck = Object.keys(extendedUpdate);
        if (toCheck.length > 0) {
          const existing = await this._getExistingPatientColumns(prisma, toCheck);
          const existingSet = new Set(existing);
          const filtered = {};
          for (const k of toCheck) if (existingSet.has(k)) filtered[k] = extendedUpdate[k];
          if (Object.keys(filtered).length > 0) {
            await prisma.patients.update({ where: { id: created.id }, data: filtered });
          }
        }
      } catch (err) {
        console.warn('Failed to apply extended fields to dependent after create:', err.message || err);
      }

      return created;
    } catch (error) {
      console.error('Error in createDependentPatient:', error);
      throw error;
    }
  }

  // Create a patient row without creating a linked user account.
  // Used by reception when they add an ad-hoc patient so we don't reserve an email/password.
  async createPatientWithoutUser(patientData) {
    try {
      const {
        full_name,
        phone,
        dob,
        gender,
        blood_type,
        allergies,
        email,
        address,
        emergency_contact
      } = patientData;
      // Default password for receptionist-created accounts
      const DEFAULT_PASSWORD = 'patient123';

      // Check if a patient with same phone or email exists to avoid duplicates
      const existingPatient = (phone || email)
        ? await prisma.patients.findFirst({
            where: {
              OR: [phone ? { phone } : undefined, email ? { email } : undefined].filter(Boolean)
            }
          })
        : null;

      if (existingPatient) {
        console.warn('createPatientWithoutUser: patient already exists, returning existing id=', existingPatient.id);
        return existingPatient;
      }

      // Use a transaction: create or reuse user, then create patient linked to that user
      const result = await prisma.$transaction(async (tx) => {
        // Find patient role
        const patientRole = await tx.roles.findUnique({ where: { name: ROLES.PATIENT } });
        if (!patientRole) throw new Error('Patient role not found');

        // Try to find existing user by email (preferred) or phone
        let user = null;
        if (email) {
          user = await tx.users.findUnique({ where: { email } });
        }
        if (!user && phone) {
          user = await tx.users.findFirst({ where: { phone } });
        }

        let tempPassword = null;
        if (!user) {
          // create new user with default password so patient can login/receive notifications
          const hashed = await hashPassword(DEFAULT_PASSWORD);
          const userData = {
            email: email || null,
            password: hashed,
            full_name: full_name || (email ? email.split('@')[0] : 'Patient'),
            phone: phone || null,
            dob: dob ? new Date(dob) : null,
            role_id: patientRole.id,
            is_active: true
          };
          user = await tx.users.create({ data: userData });
          tempPassword = DEFAULT_PASSWORD;
        }

        const patientCreateData = {
          user_id: user.id,
          full_name: full_name || null,
          phone: phone || null,
          dob: dob ? new Date(dob) : null,
          gender: gender || null,
          blood_type: blood_type || null,
          allergies: allergies || null,
          email: email || null,
          address: address || null,
          emergency_contact: emergency_contact || {}
        };

        const patient = await tx.patients.create({ data: patientCreateData });

        return { patient, tempPassword };
      });

      // Return patient row and temp password if a new user was created
      const createdPatient = result.patient;
      if (result.tempPassword) {
        // attach temp password for frontend to show to receptionist
        createdPatient.temp_password = result.tempPassword;
      }
      return createdPatient;
    } catch (error) {
      console.error('Error in createPatientWithoutUser:', error);
      throw error;
    }
  }

  async getPatientsByOwner(ownerUserId) {
    try {
      const patients = await prisma.patients.findMany({
        where: {
          OR: [
            { owner_user_id: ownerUserId },
            { user_id: ownerUserId }
          ]
        },
        include: {
          user: {
            select: {
              id: true,
              full_name: true,
              phone: true,
              dob: true,
              avatar_url: true,
              email: true
            }
          }
        },
        orderBy: { created_at: 'desc' }
      });

      return patients;
    } catch (error) {
      console.error('Error in getPatientsByOwner:', error);
      throw error;
    }
  }

    // Return shared_medical_records rows for all patient rows owned by this user
    async getSharedMedicalRecordsForUser(userId) {
      try {
        // find all patient rows related to this user (user_id or owner_user_id)
        const patients = await prisma.patients.findMany({
          where: { OR: [{ user_id: userId }, { owner_user_id: userId }] },
          select: { id: true }
        });
        const patientIds = patients.map(p => p.id).filter(Boolean);
        if (patientIds.length === 0) return [];

        const rows = await prisma.$queryRaw`
          SELECT id, medical_record_id, recipient_patient_id, shared_by_user_id, created_at
          FROM shared_medical_records
          WHERE recipient_patient_id = ANY(${patientIds})
          ORDER BY created_at DESC
        `;

        return Array.isArray(rows) ? rows : [];
      } catch (error) {
        console.error('Error in getSharedMedicalRecordsForUser:', error);
        throw error;
      }
    }

  async updatePatient(id, updateData) {
    try {
      const patient = await prisma.patients.findUnique({
        where: { id },
        include: { user: true }  // ← Fixed: users → user
      });

      if (!patient) {
        return null;
      }

      const result = await prisma.$transaction(async (tx) => {
        // Update user info
        const userUpdateData = {};
        if (updateData.full_name) userUpdateData.full_name = updateData.full_name;
        if (updateData.phone) userUpdateData.phone = updateData.phone;
        if (updateData.email) userUpdateData.email = updateData.email;
        if (updateData.dob) userUpdateData.dob = new Date(updateData.dob);

        if (Object.keys(userUpdateData).length > 0) {
          await tx.users.update({
            where: { id: patient.user_id },
            data: userUpdateData
          });
        }

        // Update patient info
        const patientUpdateData = {};
        if (updateData.gender) patientUpdateData.gender = updateData.gender;
        if (updateData.blood_type) patientUpdateData.blood_type = updateData.blood_type;
        if (updateData.allergies !== undefined) patientUpdateData.allergies = updateData.allergies;
        if (updateData.emergency_contact) patientUpdateData.emergency_contact = updateData.emergency_contact;

        // filter patientUpdateData to only include columns that exist
        const toCheck = Object.keys(patientUpdateData);
        let filteredPatientUpdateData = patientUpdateData;
        if (toCheck.length > 0) {
          const existingCols = await this._getExistingPatientColumns(tx, toCheck);
          const existingSet = new Set(existingCols);
          filteredPatientUpdateData = {};
          for (const k of Object.keys(patientUpdateData)) {
            if (existingSet.has(k)) filteredPatientUpdateData[k] = patientUpdateData[k];
          }
        }

        const updatedPatient = await tx.patients.update({
          where: { id },
          data: filteredPatientUpdateData,
          include: {
            user: {  // ← Fixed: users → user
              select: {
                id: true,
                full_name: true,
                email: true,
                phone: true,
                dob: true
              }
            }
          }
        });

        return updatedPatient;
      });

      return result;
    } catch (error) {
      console.error('Error in updatePatient:', error);
      throw error;
    }
  }

  async updatePatientProfile(userId, updateData) {
    try {
      const patient = await prisma.patients.findFirst({
        where: { user_id: userId }
      });

      if (!patient) {
        return null;
      }

      const result = await prisma.$transaction(async (tx) => {
        // Update user info
        const userUpdateData = {};
        if (updateData.full_name) userUpdateData.full_name = updateData.full_name;
        if (updateData.phone) userUpdateData.phone = updateData.phone;
        if (updateData.dob) userUpdateData.dob = updateData.dob;
        if (updateData.avatar_url !== undefined) userUpdateData.avatar_url = updateData.avatar_url;

        if (Object.keys(userUpdateData).length > 0) {
          await tx.users.update({
            where: { id: userId },
            data: userUpdateData
          });
        }

        // Update patient info - include extended fields
        const patientUpdateData = {};
        if (updateData.gender) patientUpdateData.gender = updateData.gender;
        if (updateData.address) patientUpdateData.address = updateData.address;
        if (updateData.emergency_contact) patientUpdateData.emergency_contact = updateData.emergency_contact;
        if (updateData.occupation !== undefined) patientUpdateData.occupation = updateData.occupation;
        if (updateData.id_type !== undefined) patientUpdateData.id_type = updateData.id_type;
        if (updateData.id_number !== undefined) patientUpdateData.id_number = updateData.id_number;
        if (updateData.nationality !== undefined) patientUpdateData.nationality = updateData.nationality;
        if (updateData.ethnicity !== undefined) patientUpdateData.ethnicity = updateData.ethnicity;
        // old/new address fields removed - we use single 'address' field

        // filter patientUpdateData to only include existing columns
        const toCheck = Object.keys(patientUpdateData);
        let filteredPatientUpdateData = patientUpdateData;
        if (toCheck.length > 0) {
          const existingCols = await this._getExistingPatientColumns(tx, toCheck);
          const existingSet = new Set(existingCols);
          filteredPatientUpdateData = {};
          for (const k of Object.keys(patientUpdateData)) {
            if (existingSet.has(k)) filteredPatientUpdateData[k] = patientUpdateData[k];
            else console.warn(`Skipping update of patient column '${k}' because it does not exist in DB`);
          }
        }

        const updatedPatient = await tx.patients.update({
          where: { id: patient.id },
          data: filteredPatientUpdateData,
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                email: true,
                phone: true,
                avatar_url: true,
                dob: true
              }
            }
          }
        });

        return updatedPatient;
      });

      return result;
    } catch (error) {
      console.error('Error in updatePatientProfile:', error);
      throw error;
    }
  }

  async updateDependentPatient(ownerUserId, patientId, updateData) {
    try {
      const patient = await prisma.patients.findUnique({ where: { id: patientId } });

      if (!patient) return null;

      // Check ownership: either the patient is linked to the user, or owner_user_id matches
      if (patient.user_id !== ownerUserId && patient.owner_user_id !== ownerUserId) {
        const err = new Error('Not authorized to update this patient');
        err.status = 403;
        throw err;
      }

      const result = await prisma.$transaction(async (tx) => {
        // Update linked user if exists
        const userUpdateData = {};
        if (updateData.full_name) userUpdateData.full_name = updateData.full_name;
        if (updateData.phone) userUpdateData.phone = updateData.phone;
        if (updateData.email) userUpdateData.email = updateData.email;
        if (updateData.dob) {
          // normalize dob to Date object (Prisma expects Date for Date fields)
          const d = new Date(updateData.dob);
          if (!isNaN(d)) userUpdateData.dob = d;
        }
        if (updateData.avatar_url !== undefined && patient.user_id) userUpdateData.avatar_url = updateData.avatar_url;

        if (patient.user_id && Object.keys(userUpdateData).length > 0) {
          await tx.users.update({ where: { id: patient.user_id }, data: userUpdateData });
        }

        // Update patient record (extended fields)
        const patientUpdateData = {};
        if (updateData.gender) patientUpdateData.gender = updateData.gender;
        if (updateData.address) patientUpdateData.address = updateData.address;
        if (updateData.emergency_contact) {
          // ensure emergency_contact is JSON/object
          patientUpdateData.emergency_contact = typeof updateData.emergency_contact === 'string'
            ? (updateData.emergency_contact ? JSON.parse(updateData.emergency_contact) : {})
            : updateData.emergency_contact;
        }
        if (updateData.occupation !== undefined) patientUpdateData.occupation = updateData.occupation;
        if (updateData.id_type !== undefined) patientUpdateData.id_type = updateData.id_type;
        if (updateData.id_number !== undefined) patientUpdateData.id_number = updateData.id_number;
        if (updateData.id_issue_date !== undefined) {
          const idd = updateData.id_issue_date ? new Date(updateData.id_issue_date) : null;
          if (idd && !isNaN(idd)) patientUpdateData.id_issue_date = idd;
          else if (idd === null) patientUpdateData.id_issue_date = null;
        }
        if (updateData.id_issue_place !== undefined) patientUpdateData.id_issue_place = updateData.id_issue_place;
        if (updateData.nationality !== undefined) patientUpdateData.nationality = updateData.nationality;
        if (updateData.ethnicity !== undefined) patientUpdateData.ethnicity = updateData.ethnicity;
        // old/new address fields removed - we use single 'address' field
        if (updateData.zalo !== undefined) patientUpdateData.zalo = updateData.zalo;

        // Ensure we only try to update columns that actually exist in the patients table
        const toCheck = Object.keys(patientUpdateData);
        let filteredPatientUpdateData = patientUpdateData;
        if (toCheck.length > 0) {
          const colsCondition2 = toCheck.map(c => `'${c}'`).join(',');
          const existingColsRows = await tx.$queryRawUnsafe(`SELECT column_name FROM information_schema.columns WHERE table_name = 'patients' AND column_name IN (${colsCondition2})`);
          const existingColsSet = new Set(Array.isArray(existingColsRows) ? existingColsRows.map(r => r.column_name) : []);
          filteredPatientUpdateData = {};
          for (const k of Object.keys(patientUpdateData)) {
            if (existingColsSet.has(k)) filteredPatientUpdateData[k] = patientUpdateData[k];
            else {
              // column missing in DB schema; skip updating it
              console.warn(`Skipping update of patient column '${k}' because it does not exist in DB`);
            }
          }
        }

        const updatedPatient = await tx.patients.update({
          where: { id: patientId },
          data: filteredPatientUpdateData,
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                email: true,
                phone: true,
                avatar_url: true,
                dob: true
              }
            }
          }
        });

        return updatedPatient;
      });

      return result;
    } catch (error) {
      console.error('Error in updateDependentPatient:', error);
      throw error;
    }
  }

  async deletePatient(id) {
    try {
      const patient = await prisma.patients.findUnique({
        where: { id }
      });

      if (!patient) {
        throw new Error('Patient not found');
      }

      const activeAppointments = await prisma.appointments.count({
        where: {
          patient_id: id,
          appointment_date: { gte: new Date() },
          status: { in: ['PENDING', 'CONFIRMED', 'CHECKED_IN'] }
        }
      });

      if (activeAppointments > 0) {
        throw new Error('Cannot delete patient with active appointments');
      }

      // If this patient is linked to a user account, delete the user (will cascade delete patient if FK cascade configured).
      // Otherwise, manually delete the patient row.
      if (patient.user_id) {
        await prisma.$transaction(async (tx) => {
          await tx.users.delete({ where: { id: patient.user_id } });
        });
      } else {
        await prisma.patients.delete({ where: { id } });
      }
    } catch (error) {
      console.error('Error in deletePatient:', error);
      throw error;
    }
  }
}

module.exports = new PatientsService();