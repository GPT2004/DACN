/* eslint-disable no-unused-vars */
import React, { useRef, useState } from 'react';
import Hero from '../../components/patient/clinic/Hero';
import Services from '../../components/patient/clinic/Services';
import Doctors from '../../components/patient/clinic/Doctors';
import Contact from '../../components/patient/clinic/Contact';
import Footer from '../../components/patient/clinic/Footer';

export default function PatientDashboard() {
	const [selectedSpecialty, setSelectedSpecialty] = useState(null);

	const handleBookNow = (doctor) => {
		// Navigate to appointments booking if needed
	};

	return (
		<div className="min-h-screen bg-gray-50">
			<Hero onBook={handleBookNow} />
			<main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-12">
				<div className="bg-white rounded-lg shadow-lg p-6">
					<Services onSelectSpecialty={setSelectedSpecialty} />
				</div>

				<div className="mt-8">
					<div className="bg-white rounded-lg shadow p-6">
						<h2 className="text-xl font-semibold mb-4">Bác sĩ nổi bật</h2>
						<Doctors onBook={handleBookNow} specialty={selectedSpecialty} />
					</div>
				</div>

				<div className="mt-8 bg-white rounded-lg shadow p-6">
					<Contact />
				</div>
			</main>

			<Footer />
		</div>
	);
}
