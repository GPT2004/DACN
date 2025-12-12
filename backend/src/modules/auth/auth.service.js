const prisma = require('../../config/database');
const { hashPassword, comparePassword } = require('../../utils/bcrypt');
const { generateToken, generateRefreshToken } = require('../../utils/jwt');
const { ROLES } = require('../../config/constants');
const crypto = require('crypto');
const emailUtil = require('../../utils/email');

const DEFAULT_TOKEN_EXPIRY_MINUTES = parseInt(process.env.REGISTRATION_TOKEN_EXPIRY_MINUTES || '60', 10);

class AuthService {
  async register(userData) {
    // Create a pending registration and send confirmation email instead of creating account immediately
    const existingUser = await prisma.users.findUnique({ where: { email: userData.email } });
    if (existingUser) {
      throw new Error('Email already registered');
    }

    // prepare token and expiry
    const token = crypto.randomBytes(32).toString('hex');
    const expiresAt = new Date(Date.now() + DEFAULT_TOKEN_EXPIRY_MINUTES * 60 * 1000);

    const hashedPassword = await hashPassword(userData.password);

    // store other profile data in data JSON
    const extraData = {
      phone: userData.phone,
      dob: userData.dob || null,
      gender: userData.gender || null,
      blood_type: userData.blood_type || null,
      allergies: userData.allergies || null,
    };

    // insert pending registration via raw SQL (Prisma model not added to schema)
    // Ensure the pending_registrations table exists (avoid migration requirement)
    try {
      await prisma.$executeRaw`
        CREATE TABLE IF NOT EXISTS "pending_registrations" (
          id SERIAL PRIMARY KEY,
          email VARCHAR(255) NOT NULL UNIQUE,
          full_name VARCHAR(255),
          password_hash TEXT NOT NULL,
          data JSONB DEFAULT '{}',
          token VARCHAR(255) NOT NULL UNIQUE,
          expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
      `;
    } catch (ensureErr) {
      // If this fails, continue and let the insert fail with a clear error
      console.warn('Could not ensure pending_registrations table exists:', ensureErr.message || ensureErr);
    }

    const inserted = await prisma.$queryRaw`
      INSERT INTO pending_registrations (email, full_name, password_hash, data, token, expires_at, created_at)
      VALUES (
        ${userData.email},
        ${userData.full_name},
        ${hashedPassword},
        ${JSON.stringify(extraData)}::jsonb,
        ${token},
        ${expiresAt},
        now()
      )
      RETURNING *;
    `;
    // inserted may be array-like; keep first row
    const pendingRow = Array.isArray(inserted) ? inserted[0] : inserted;

    // send confirmation email
    const frontendUrl = process.env.FRONTEND_URL;
    const apiBase = process.env.API_BASE_URL || `http://localhost:${process.env.PORT || 3000}`;
    let confirmUrl;
    if (frontendUrl) {
      confirmUrl = `${frontendUrl.replace(/\/$/, '')}/confirm-registration?token=${token}`;
    } else {
      // fallback: call backend confirm endpoint directly
      confirmUrl = `${apiBase.replace(/\/$/, '')}/api/auth/confirm-registration?token=${token}`;
    }

    const html = `
      <p>Xin chào ${userData.full_name || ''},</p>
      <p>Cảm ơn bạn đã đăng ký. Vui lòng nhấp vào liên kết bên dưới để xác nhận email và hoàn tất đăng ký:</p>
      <p><a href="${confirmUrl}">Xác nhận đăng ký</a></p>
      <p>Liên kết này có hiệu lực trong ${DEFAULT_TOKEN_EXPIRY_MINUTES} phút.</p>
    `;

    try {
      await emailUtil.sendEmail({ to: userData.email, subject: 'Xác nhận đăng ký - Phòng khám', html });
    } catch (err) {
      // If email fails, still keep pending registration but log error (email util logs internally)
    }

    return { message: 'Registration pending. Please check your email to confirm.' };
  }

  async confirmRegistration(token) {
    // find pending registration
    // fetch pending registration using raw query
    const pendingRows = await prisma.$queryRaw`
      SELECT * FROM pending_registrations WHERE token = ${token} LIMIT 1
    `;
    const pending = Array.isArray(pendingRows) && pendingRows.length > 0 ? pendingRows[0] : null;
    if (!pending) {
      throw new Error('Token invalid or not found');
    }

    if (new Date(pending.expires_at) < new Date()) {
      // delete expired pending
      await prisma.$queryRaw`DELETE FROM pending_registrations WHERE id = ${pending.id}`;
      throw new Error('Token expired');
    }

    // ensure email still not registered
    const existingUser = await prisma.users.findUnique({ where: { email: pending.email } });
    if (existingUser) {
      // cleanup pending
      await prisma.$queryRaw`DELETE FROM pending_registrations WHERE id = ${pending.id}`;
      throw new Error('Email already registered');
    }

    // Get patient role
    const patientRole = await prisma.roles.findUnique({ where: { name: ROLES.PATIENT } });
    if (!patientRole) throw new Error('Patient role not found');

    // Create user and patient in transaction
    const result = await prisma.$transaction(async (tx) => {
      const user = await tx.users.create({
        data: {
          email: pending.email,
          password: pending.password_hash,
          full_name: pending.full_name,
          phone: pending.data.phone,
          dob: pending.data.dob,
          role_id: patientRole.id,
        },
        include: { role: true },
      });

      const patient = await tx.patients.create({
        data: {
          user_id: user.id,
          gender: pending.data.gender,
          blood_type: pending.data.blood_type,
          allergies: pending.data.allergies,
        },
      });

      // delete pending row
      await tx.$queryRaw`DELETE FROM pending_registrations WHERE id = ${pending.id}`;

      return { user, patient };
    });

    // Generate tokens
    const tokenJwt = generateToken({ userId: result.user.id });
    const refreshToken = generateRefreshToken({ userId: result.user.id });

    return {
      user: this.sanitizeUser(result.user),
      token: tokenJwt,
      refreshToken,
    };
  }

  // Forgot password: create a reset token and email the user a confirmation link
  async forgotPassword(email) {
    const rawEmail = (email || '').trim().toLowerCase();

    // Find user by email OR via patients table (if patient row links to a user)
    let user = await prisma.users.findUnique({ where: { email: rawEmail } });

    if (!user) {
      // try to find a patient with this email that has a linked user
      const patient = await prisma.patients.findFirst({ where: { email } });
      if (patient && patient.user_id) {
        user = await prisma.users.findUnique({ where: { id: patient.user_id } });
      }
    }

    // Always return generic message, but only create reset if a user record exists
    if (!user) {
      return { message: 'If the email exists, a confirmation was sent' };
    }

    const token = crypto.randomBytes(32).toString('hex');
    const expiresAt = new Date(Date.now() + DEFAULT_TOKEN_EXPIRY_MINUTES * 60 * 1000);

    // Ensure table exists
    try {
      await prisma.$executeRaw`
        CREATE TABLE IF NOT EXISTS "password_resets" (
          id SERIAL PRIMARY KEY,
          email VARCHAR(255) NOT NULL,
          token VARCHAR(255) NOT NULL UNIQUE,
          expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
      `;
    } catch (err) {
      console.warn('Could not ensure password_resets table exists:', err.message || err);
    }

    // Insert reset row using the resolved user's email
    const targetEmail = (user.email || '').trim().toLowerCase();
    await prisma.$queryRaw`
      INSERT INTO password_resets (email, token, expires_at, created_at)
      VALUES (${targetEmail}, ${token}, ${expiresAt}, now())
    `;

    // Build confirmation URL (prefer frontend)
    const frontendUrl = process.env.FRONTEND_URL;
    const apiBase = process.env.API_BASE_URL || `http://localhost:${process.env.PORT || 3000}`;
    let confirmUrl;
    if (frontendUrl) {
      confirmUrl = `${frontendUrl.replace(/\/$/, '')}/confirm-reset?token=${token}`;
    } else {
      confirmUrl = `${apiBase.replace(/\/$/, '')}/api/auth/confirm-reset?token=${token}`;
    }

    const html = `
      <p>Xin chào ${user.full_name || ''},</p>
      <p>Bạn đã yêu cầu lấy lại mật khẩu. Vui lòng nhấp vào liên kết dưới đây để xác nhận và nhận mật khẩu mới:</p>
      <p><a href="${confirmUrl}">Xác nhận đặt lại mật khẩu</a></p>
      <p>Liên kết này có hiệu lực trong ${DEFAULT_TOKEN_EXPIRY_MINUTES} phút.</p>
    `;

    try {
      await emailUtil.sendEmail({ to: targetEmail, subject: 'Xác nhận đặt lại mật khẩu', html });
    } catch (err) {
      // ignore send errors; still return generic message
    }

    return { message: 'If the email exists, a confirmation was sent' };
  }

  // Confirm reset: verify token, generate new password, update user, email new password
  async confirmReset(token) {
    // fetch reset row
    const rows = await prisma.$queryRaw`
      SELECT * FROM password_resets WHERE token = ${token} LIMIT 1
    `;
    const reset = Array.isArray(rows) && rows.length > 0 ? rows[0] : null;
    if (!reset) throw new Error('Token invalid or not found');
    if (new Date(reset.expires_at) < new Date()) {
      await prisma.$queryRaw`DELETE FROM password_resets WHERE id = ${reset.id}`;
      throw new Error('Token expired');
    }

    const email = reset.email;
    const user = await prisma.users.findUnique({ where: { email } });
    if (!user) {
      await prisma.$queryRaw`DELETE FROM password_resets WHERE id = ${reset.id}`;
      throw new Error('User not found');
    }

    // generate a new password (12 chars hex)
    const newPasswordPlain = crypto.randomBytes(6).toString('hex');
    const hashed = await hashPassword(newPasswordPlain);

    await prisma.users.update({ where: { id: user.id }, data: { password: hashed } });

    // delete reset row
    await prisma.$queryRaw`DELETE FROM password_resets WHERE id = ${reset.id}`;

    // send email with new password
    const html = `
      <p>Xin chào ${user.full_name || ''},</p>
      <p>Mật khẩu mới của bạn là: <strong>${newPasswordPlain}</strong></p>
      <p>Vui lòng đăng nhập và thay đổi mật khẩu trong phần cài đặt để bảo mật tài khoản.</p>
    `;

    try {
      await emailUtil.sendEmail({ to: email, subject: 'Mật khẩu mới - Phòng khám', html });
    } catch (err) {
      console.warn('Failed to send new password email:', err.message || err);
    }

    // For debugging in development, optionally return the plaintext password
    if (process.env.DEBUG_PASSWORD_RESET === 'true') {
      console.warn(`DEBUG_PASSWORD_RESET: password for user ${user.id} (${email}) = ${newPasswordPlain}`);
      return { message: 'Password reset successful', newPassword: newPasswordPlain };
    }

    return { message: 'Password reset successful' };
  }

  async checkEmail(email) {
    if (!email) return false;
    const rawEmail = (email || '').trim().toLowerCase();
    const user = await prisma.users.findUnique({ where: { email: rawEmail } });
    if (user) return true;
    // If no user, check patients for an email that links to a user
    const patient = await prisma.patients.findFirst({ where: { email: rawEmail } });
    if (patient && patient.user_id) return true;
    return false;
  }

  async login(email, password) {
    const user = await prisma.users.findUnique({
      where: { email },
      include: { role: true },
    });

    if (!user) {
      throw new Error('Invalid email or password');
    }

    if (!user.is_active) {
      throw new Error('Account is inactive. Please contact administrator.');
    }

    const isPasswordValid = await comparePassword(password, user.password);

    if (!isPasswordValid) {
      throw new Error('Invalid email or password');
    }

    const token = generateToken({ userId: user.id });
    const refreshToken = generateRefreshToken({ userId: user.id });

    return {
      user: this.sanitizeUser(user),
      token,
      refreshToken,
    };
  }

  async changePassword(userId, currentPassword, newPassword) {
    const user = await prisma.users.findUnique({
      where: { id: userId },
    });

    const isPasswordValid = await comparePassword(currentPassword, user.password);

    if (!isPasswordValid) {
      throw new Error('Current password is incorrect');
    }

    const hashedPassword = await hashPassword(newPassword);

    await prisma.users.update({
      where: { id: userId },
      data: { password: hashedPassword },
    });

    return { message: 'Password changed successfully' };
  }

  async getProfile(userId) {
    const user = await prisma.users.findUnique({
      where: { id: userId },
      include: {
        role: true,
        patients: true,
        doctors: true,
      },
    });

    if (!user) {
      throw new Error('User not found');
    }

    return this.sanitizeUser(user);
  }

  sanitizeUser(user) {
    const { ...userWithoutPassword } = user;
    return userWithoutPassword;
  }
}

module.exports = new AuthService();