const calculateAge = (dob) => {
  const today = new Date();
  const birthDate = new Date(dob);
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  
  return age;
};

const formatDate = (date, format = 'YYYY-MM-DD') => {
  const d = new Date(date);
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  
  return format
    .replace('YYYY', year)
    .replace('MM', month)
    .replace('DD', day);
};

const generateCode = (prefix, length = 6) => {
  const randomNum = Math.floor(Math.random() * Math.pow(10, length));
  return `${prefix}${String(randomNum).padStart(length, '0')}`;
};

const sanitizeObject = (obj, allowedFields) => {
  return Object.keys(obj)
    .filter(key => allowedFields.includes(key))
    .reduce((acc, key) => {
      acc[key] = obj[key];
      return acc;
    }, {});
};

module.exports = {
  calculateAge,
  formatDate,
  generateCode,
  sanitizeObject,
};