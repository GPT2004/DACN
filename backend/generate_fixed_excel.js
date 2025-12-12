const XLSX = require('xlsx');
const path = require('path');

const fixedData = [
  { name: 'Paracetamol', code: 'PARA001', quantity: 100, expiry_date: '31/12/2026', batch_number: 'LOT2026001', unit: 'viên', price: 500, description: 'Thuốc giảm đau' },
  { name: 'Vitamin C', code: 'VITC001', quantity: 50, expiry_date: '15/06/2026', batch_number: 'LOT2026002', unit: 'hộp', price: 25000, description: 'Vitamin bổ sung' },
  { name: 'Amoxicillin', code: 'AMOX001', quantity: 75, expiry_date: '30/11/2026', batch_number: 'LOT2026003', unit: 'viên', price: 800, description: 'Kháng sinh Amoxicillin' },
  { name: 'Ibuprofen', code: 'IBUP001', quantity: 80, expiry_date: '20/08/2026', batch_number: 'LOT2026004', unit: 'viên', price: 600, description: 'Thuốc chống viêm' },
  { name: 'Aspirin', code: 'ASPI001', quantity: 120, expiry_date: '10/03/2027', batch_number: 'LOT2026005', unit: 'viên', price: 300, description: 'Thuốc chống đông máu' }
];

const wb = XLSX.utils.book_new();
const ws = XLSX.utils.json_to_sheet(fixedData, { header: ['name','code','quantity','expiry_date','batch_number','unit','price','description'] });
ws['!cols'] = [{wch:15},{wch:10},{wch:8},{wch:12},{wch:12},{wch:8},{wch:10},{wch:25}];
XLSX.utils.book_append_sheet(wb, ws, 'Thuoc');
const outPath = path.join(__dirname, 'test_medicines_fixed.xlsx');
XLSX.writeFile(wb, outPath);
console.log('Created', outPath);
