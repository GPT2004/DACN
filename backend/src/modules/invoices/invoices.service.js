const prisma = require('../../config/database');
const { ROLES, INVOICE_STATUS } = require('../../config/constants');
const notificationsService = require('../notifications/notifications.service');
const { PRESCRIPTION_STATUS } = require('../../config/constants');

class InvoiceService {
  async createInvoice(data) {
    // Verify appointment and patient
    let patient_id = data.patient_id;
    
    if (data.appointment_id) {
      const appointment = await prisma.appointments.findUnique({
        where: { id: parseInt(data.appointment_id) },
        include: {
          doctor: true,
        },
      });

      if (!appointment) {
        throw new Error('Appointment not found');
      }

      patient_id = appointment.patient_id;
    }

    // Verify patient exists
    const patient = await prisma.patients.findUnique({
      where: { id: parseInt(patient_id) },
    });

    if (!patient) {
      throw new Error('Patient not found');
    }

    // Calculate totals
    let subtotal = 0;
    const items = data.items.map(item => {
      const itemTotal = item.unit_price * item.quantity;
      subtotal += itemTotal;
      return {
        type: item.type || 'other',
        description: item.description,
        quantity: item.quantity,
        unit_price: item.unit_price,
        amount: itemTotal,
      };
    });

    // Per product decision: omit tax/VAT (tax = 0)
    const tax = 0;
    const discount = data.discount || 0;
    const total = subtotal + tax - discount;

    // Create invoice
    const invoice = await prisma.invoices.create({
      data: {
        appointment_id: data.appointment_id ? parseInt(data.appointment_id) : null,
        patient_id: parseInt(patient_id),
        // Ensure item shape includes description, quantity, unit_price and amount
        items: items,
        subtotal: subtotal,
        tax: tax,
        discount: discount,
        total: total,
        status: INVOICE_STATUS.UNPAID,
      },
      include: {
        patient: {
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                phone: true,
                email: true,
              },
            },
          },
        },
        appointment: {
          include: {
            doctor: {
              include: {
                user: {
                  select: {
                    full_name: true,
                  },
                },
              },
            },
          },
        },
      },
    });

    return invoice;
  }

  async createInvoiceFromPrescription(prescription_id, created_by = null) {
    const prescription = await prisma.prescriptions.findUnique({
      where: { id: parseInt(prescription_id) },
      include: {
        patient: true,
        doctor: true,
        appointment: {
          include: {
            doctor: true,
          },
        },
      },
    });

    if (!prescription) {
      throw new Error('Prescription not found');
    }

    // Derive invoice items from prescription items
    const prescItems = typeof prescription.items === 'string' ? JSON.parse(prescription.items) : prescription.items;

    let subtotal = 0;
    const items = prescItems.map(item => {
      const unit_price = item.unit_price || 0;
      const quantity = item.quantity || 1;
      const amount = unit_price * quantity;
      subtotal += amount;
      return {
        type: 'medicine',
        description: item.medicine_name || item.description || 'Thuốc',
        quantity: quantity,
        unit_price: unit_price,
        amount: amount,
      };
    });

    // If doctor's consultation fee is available (from appointment or prescription), add it as an item
    try {
      const consultFee = prescription?.appointment?.doctor?.consultation_fee || prescription?.doctor?.consultation_fee || 0;
      if (consultFee && Number(consultFee) > 0) {
        const consultAmount = Number(consultFee) || 0;
        items.unshift({ type: 'consultation', description: 'Phí khám', quantity: 1, unit_price: consultAmount, amount: consultAmount });
        subtotal += consultAmount;
      }
    } catch (e) {
      // safe fallback: ignore if nested fields missing
    }

    const tax = 0;
    const discount = 0;
    const total = subtotal + tax - discount;

    const invoice = await prisma.invoices.create({
      data: {
        appointment_id: prescription.appointment_id || null,
        patient_id: prescription.patient_id,
        prescription_id: prescription.id,
        items: items,
        subtotal: subtotal,
        tax: tax,
        discount: discount,
        total: total,
        status: INVOICE_STATUS.UNPAID,
      },
      include: {
        patient: {
          include: { user: true },
        },
        appointment: true,
      },
    });

    // mark prescription as INVOICED so receptionist->pharmacist flow is clearer
    try {
      await prisma.prescriptions.update({ where: { id: parseInt(prescription.id) }, data: { status: PRESCRIPTION_STATUS.INVOICED, updated_at: new Date() } });
    } catch (err) {
      console.error('Failed to update prescription status to INVOICED:', err.message || err);
    }

    return invoice;
  }

  async getInvoices(filters, user) {
    const { patient_id, status, start_date, end_date, page, limit } = filters;
    const skip = (page - 1) * limit;

    const where = {};

    // Role-based filtering
    if (user.role.name === ROLES.PATIENT) {
      const patient = await prisma.patients.findFirst({
        where: { user_id: user.id },
      });
      if (patient) {
        where.patient_id = patient.id;
      }
    } else if (patient_id) {
      where.patient_id = parseInt(patient_id);
    }

    if (status) {
      where.status = status;
    }

    if (start_date || end_date) {
      where.created_at = {};
      if (start_date) where.created_at.gte = new Date(start_date);
      if (end_date) {
        const endDateTime = new Date(end_date);
        endDateTime.setHours(23, 59, 59, 999);
        where.created_at.lte = endDateTime;
      }
    }

    const [invoices, total] = await Promise.all([
      prisma.invoices.findMany({
        where,
        skip,
        take: limit,
        include: {
          patient: {
            include: {
              user: {
                select: {
                  id: true,
                  full_name: true,
                  phone: true,
                },
              },
            },
          },
          appointment: {
            include: {
              doctor: {
                include: {
                  user: {
                    select: {
                      full_name: true,
                    },
                  },
                },
              },
            },
          },
        },
        orderBy: {
          created_at: 'desc',
        },
      }),
      prisma.invoices.count({ where }),
    ]);

    return {
      invoices,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  async getInvoiceById(id, user) {
    const invoice = await prisma.invoices.findUnique({
      where: { id: parseInt(id) },
      include: {
        patient: {
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                phone: true,
                email: true,
                dob: true,
              },
            },
          },
        },
        appointment: {
          include: {
            doctor: {
              include: {
                user: {
                  select: {
                    id: true,
                    full_name: true,
                  },
                },
              },
            },
          },
        },
      },
    });

    if (!invoice) {
      throw new Error('Invoice not found');
    }

    // Authorization check
    if (user.role.name === ROLES.PATIENT) {
      const patient = await prisma.patients.findFirst({
        where: { user_id: user.id },
      });
      if (!patient || invoice.patient_id !== patient.id) {
        throw new Error('Unauthorized access');
      }
    }

    return invoice;
  }

  async updateInvoice(id, data) {
    const invoice = await prisma.invoices.findUnique({
      where: { id: parseInt(id) },
    });

    if (!invoice) {
      throw new Error('Invoice not found');
    }

    // Can only update UNPAID invoices
    if (invoice.status !== INVOICE_STATUS.UNPAID) {
      throw new Error('Can only update unpaid invoices');
    }

    const updateData = {};

    if (data.items) {
      let subtotal = 0;
      const items = data.items.map(item => {
        const itemTotal = item.unit_price * item.quantity;
        subtotal += itemTotal;
        return {
          type: item.type || 'other',
          description: item.description,
          quantity: item.quantity,
          unit_price: item.unit_price,
          amount: itemTotal,
        };
      });

      // No tax by requirement
      const tax = 0;
      const discount = data.discount !== undefined ? data.discount : invoice.discount;
      const total = subtotal + tax - discount;

      updateData.items = items;
      updateData.subtotal = subtotal;
      updateData.tax = tax;
      updateData.discount = discount;
      updateData.total = total;
    } else {
      if (data.discount !== undefined) {
        updateData.discount = data.discount;
        updateData.total = invoice.subtotal + invoice.tax - data.discount;
      }
    }

    updateData.updated_at = new Date();

    const updated = await prisma.invoices.update({
      where: { id: parseInt(id) },
      data: updateData,
      include: {
        patient: {
          include: {
            user: {
              select: {
                full_name: true,
              },
            },
          },
        },
      },
    });

    return updated;
  }

  async payInvoice(id, payment_method, paid_amount) {
    const invoice = await prisma.invoices.findUnique({
      where: { id: parseInt(id) },
    });

    if (!invoice) {
      throw new Error('Invoice not found');
    }

    if (invoice.status === INVOICE_STATUS.PAID) {
      throw new Error('Invoice already paid');
    }

    if (invoice.status === INVOICE_STATUS.REFUNDED) {
      throw new Error('Cannot pay refunded invoice');
    }

    const amount = paid_amount || invoice.total;

    if (amount < invoice.total) {
      throw new Error('Paid amount is less than invoice total');
    }

    const updated = await prisma.invoices.update({
      where: { id: parseInt(id) },
      data: {
        status: INVOICE_STATUS.PAID,
        paid_at: new Date(),
        updated_at: new Date(),
      },
      include: {
        patient: {
          include: {
            user: true,
          },
        },
      },
    });

    const change = amount - invoice.total;

    // If this invoice was created from a prescription, mark prescription READY_FOR_DISPENSE and notify pharmacists
    try {
      const inv = updated;
      if (inv && inv.prescription_id) {
        // Temporary hotfix: set INVOICED instead of READY_FOR_DISPENSE to avoid
        // writing an enum value that may not exist in the DB. Apply migration
        // to add READY_FOR_DISPENSE to the DB enum and restore expected flow.
        await prisma.prescriptions.update({ where: { id: parseInt(inv.prescription_id) }, data: { status: PRESCRIPTION_STATUS.INVOICED, updated_at: new Date() } });

        // broadcast notification to pharmacists
        const payload = {
          prescription_id: inv.prescription_id,
          invoice_id: inv.id,
          patient_id: inv.patient_id,
          patient_user_id: inv.patient?.user?.id || null,
          patient_name: inv.patient?.user?.full_name || inv.patient?.full_name || null,
          patient_phone: inv.patient?.user?.phone || null,
          message: `Prescription #${inv.prescription_id} is ready for dispensing (Invoice #${inv.id}).`,
        };
        await notificationsService.broadcastNotification({ role_name: 'Pharmacist', type: 'prescription_ready', payload });
      }
    } catch (err) {
      console.error('Error updating prescription after payment or sending notification:', err.message || err);
    }

    return { invoice: updated, change };
  }

  async refundInvoice(id) {
    const invoice = await prisma.invoices.findUnique({
        where: { id: parseInt(id) },
    });

    if (!invoice) {
        throw new Error('Invoice not found');
    }

    if (invoice.status !== INVOICE_STATUS.PAID) {
        throw new Error('Only paid invoices can be refunded');
    }

    const updated = await prisma.invoices.update({
        where: { id: parseInt(id) },
        data: {
        status: INVOICE_STATUS.REFUNDED,
        updated_at: new Date(),
        },
    });

    return updated;
}

  async deleteInvoice(id) {
    const invoice = await prisma.invoices.findUnique({
      where: { id: parseInt(id) },
    });

    if (!invoice) {
      throw new Error('Invoice not found');
    }

    // Can only delete UNPAID invoices
    if (invoice.status !== INVOICE_STATUS.UNPAID) {
      throw new Error('Can only delete unpaid invoices');
    }

    await prisma.invoices.delete({
      where: { id: parseInt(id) },
    });

    return { message: 'Invoice deleted successfully' };
  }

  async getRevenueSummary(start_date, end_date) {
    const where = {
      status: INVOICE_STATUS.PAID,
    };

    if (start_date || end_date) {
      where.paid_at = {};
      if (start_date) where.paid_at.gte = new Date(start_date);
      if (end_date) {
        const endDateTime = new Date(end_date);
        endDateTime.setHours(23, 59, 59, 999);
        where.paid_at.lte = endDateTime;
      }
    }

    const [totalRevenue, invoiceCount, averageInvoice] = await Promise.all([
      prisma.invoices.aggregate({
        where,
        _sum: {
          total: true,
          subtotal: true,
          tax: true,
          discount: true,
        },
      }),
      prisma.invoices.count({ where }),
      prisma.invoices.aggregate({
        where,
        _avg: {
          total: true,
        },
      }),
    ]);

    // Get unpaid invoices
    const unpaidCount = await prisma.invoices.count({
      where: {
        status: INVOICE_STATUS.UNPAID,
      },
    });

    const unpaidTotal = await prisma.invoices.aggregate({
      where: {
        status: INVOICE_STATUS.UNPAID,
      },
      _sum: {
        total: true,
      },
    });

    return {
      paid_invoices: invoiceCount,
      total_revenue: totalRevenue._sum.total || 0,
      total_subtotal: totalRevenue._sum.subtotal || 0,
      total_tax: totalRevenue._sum.tax || 0,
      total_discount: totalRevenue._sum.discount || 0,
      average_invoice: Math.round(averageInvoice._avg.total || 0),
      unpaid_invoices: unpaidCount,
      unpaid_total: unpaidTotal._sum.total || 0,
      period: {
        start_date: start_date || null,
        end_date: end_date || null,
      },
    };
  }
}

module.exports = new InvoiceService();