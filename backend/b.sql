--
-- PostgreSQL database dump
--

-- Dumped from database version 17.5
-- Dumped by pg_dump version 17.5

-- Started on 2025-11-22 00:15:16

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 6 (class 2615 OID 31206)
-- Name: public; Type: SCHEMA; Schema: -; Owner: postgres
--

-- *not* creating schema, since initdb creates it


ALTER SCHEMA public OWNER TO postgres;

--
-- TOC entry 5251 (class 0 OID 0)
-- Dependencies: 6
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON SCHEMA public IS '';


--
-- TOC entry 2 (class 3079 OID 41682)
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- TOC entry 5253 (class 0 OID 0)
-- Dependencies: 2
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- TOC entry 926 (class 1247 OID 31217)
-- Name: appointment_status; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.appointment_status AS ENUM (
    'PENDING',
    'CONFIRMED',
    'CHECKED_IN',
    'IN_PROGRESS',
    'COMPLETED',
    'CANCELLED',
    'NO_SHOW'
);


ALTER TYPE public.appointment_status OWNER TO postgres;

--
-- TOC entry 929 (class 1247 OID 31232)
-- Name: invoice_status; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.invoice_status AS ENUM (
    'UNPAID',
    'PAID',
    'REFUNDED'
);


ALTER TYPE public.invoice_status OWNER TO postgres;

--
-- TOC entry 932 (class 1247 OID 31240)
-- Name: prescription_status; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.prescription_status AS ENUM (
    'DRAFT',
    'APPROVED',
    'DISPENSED',
    'INVOICED'
);


ALTER TYPE public.prescription_status OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 218 (class 1259 OID 31207)
-- Name: _prisma_migrations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public._prisma_migrations (
    id character varying(36) NOT NULL,
    checksum character varying(64) NOT NULL,
    finished_at timestamp with time zone,
    migration_name character varying(255) NOT NULL,
    logs text,
    rolled_back_at timestamp with time zone,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    applied_steps_count integer DEFAULT 0 NOT NULL
);


ALTER TABLE public._prisma_migrations OWNER TO postgres;

--
-- TOC entry 238 (class 1259 OID 31343)
-- Name: appointments; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.appointments (
    id integer NOT NULL,
    patient_id integer NOT NULL,
    doctor_id integer NOT NULL,
    timeslot_id integer,
    appointment_date date NOT NULL,
    appointment_time time(6) without time zone,
    status public.appointment_status DEFAULT 'PENDING'::public.appointment_status NOT NULL,
    reason text,
    source character varying(50) DEFAULT 'web'::character varying,
    created_by integer,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.appointments OWNER TO postgres;

--
-- TOC entry 237 (class 1259 OID 31342)
-- Name: appointments_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.appointments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.appointments_id_seq OWNER TO postgres;

--
-- TOC entry 5254 (class 0 OID 0)
-- Dependencies: 237
-- Name: appointments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.appointments_id_seq OWNED BY public.appointments.id;


--
-- TOC entry 256 (class 1259 OID 31451)
-- Name: audit_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.audit_logs (
    id integer NOT NULL,
    user_id integer,
    action character varying(255) NOT NULL,
    meta jsonb,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.audit_logs OWNER TO postgres;

--
-- TOC entry 255 (class 1259 OID 31450)
-- Name: audit_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.audit_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.audit_logs_id_seq OWNER TO postgres;

--
-- TOC entry 5255 (class 0 OID 0)
-- Dependencies: 255
-- Name: audit_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.audit_logs_id_seq OWNED BY public.audit_logs.id;


--
-- TOC entry 258 (class 1259 OID 38293)
-- Name: doctor_reviews; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.doctor_reviews (
    id integer NOT NULL,
    doctor_id integer NOT NULL,
    patient_id integer NOT NULL,
    rating smallint NOT NULL,
    comment text,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.doctor_reviews OWNER TO postgres;

--
-- TOC entry 257 (class 1259 OID 38292)
-- Name: doctor_reviews_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.doctor_reviews_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.doctor_reviews_id_seq OWNER TO postgres;

--
-- TOC entry 5256 (class 0 OID 0)
-- Dependencies: 257
-- Name: doctor_reviews_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.doctor_reviews_id_seq OWNED BY public.doctor_reviews.id;


--
-- TOC entry 228 (class 1259 OID 31293)
-- Name: doctor_specialties; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.doctor_specialties (
    id integer NOT NULL,
    doctor_id integer NOT NULL,
    specialty_id integer NOT NULL,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.doctor_specialties OWNER TO postgres;

--
-- TOC entry 227 (class 1259 OID 31292)
-- Name: doctor_specialties_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.doctor_specialties_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.doctor_specialties_id_seq OWNER TO postgres;

--
-- TOC entry 5257 (class 0 OID 0)
-- Dependencies: 227
-- Name: doctor_specialties_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.doctor_specialties_id_seq OWNED BY public.doctor_specialties.id;


--
-- TOC entry 224 (class 1259 OID 31270)
-- Name: doctors; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.doctors (
    id integer NOT NULL,
    user_id integer NOT NULL,
    license_number character varying(100),
    specialties text[],
    bio text,
    consultation_fee integer,
    rating numeric(3,2) DEFAULT 0,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    address text,
    gender character varying(20)
);


ALTER TABLE public.doctors OWNER TO postgres;

--
-- TOC entry 223 (class 1259 OID 31269)
-- Name: doctors_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.doctors_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.doctors_id_seq OWNER TO postgres;

--
-- TOC entry 5258 (class 0 OID 0)
-- Dependencies: 223
-- Name: doctors_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.doctors_id_seq OWNED BY public.doctors.id;


--
-- TOC entry 250 (class 1259 OID 31412)
-- Name: invoices; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.invoices (
    id integer NOT NULL,
    appointment_id integer,
    patient_id integer NOT NULL,
    items jsonb NOT NULL,
    subtotal integer DEFAULT 0,
    tax integer DEFAULT 0,
    discount integer DEFAULT 0,
    total integer DEFAULT 0,
    status public.invoice_status DEFAULT 'UNPAID'::public.invoice_status NOT NULL,
    paid_at timestamp(6) with time zone,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    prescription_id integer
);


ALTER TABLE public.invoices OWNER TO postgres;

--
-- TOC entry 249 (class 1259 OID 31411)
-- Name: invoices_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.invoices_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.invoices_id_seq OWNER TO postgres;

--
-- TOC entry 5259 (class 0 OID 0)
-- Dependencies: 249
-- Name: invoices_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.invoices_id_seq OWNED BY public.invoices.id;


--
-- TOC entry 252 (class 1259 OID 31428)
-- Name: lab_orders; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.lab_orders (
    id integer NOT NULL,
    appointment_id integer,
    patient_id integer NOT NULL,
    doctor_id integer NOT NULL,
    tests jsonb NOT NULL,
    status character varying(50) DEFAULT 'PENDING'::character varying,
    results jsonb,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.lab_orders OWNER TO postgres;

--
-- TOC entry 251 (class 1259 OID 31427)
-- Name: lab_orders_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.lab_orders_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.lab_orders_id_seq OWNER TO postgres;

--
-- TOC entry 5260 (class 0 OID 0)
-- Dependencies: 251
-- Name: lab_orders_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.lab_orders_id_seq OWNED BY public.lab_orders.id;


--
-- TOC entry 240 (class 1259 OID 31356)
-- Name: medical_records; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.medical_records (
    id integer NOT NULL,
    appointment_id integer,
    patient_id integer NOT NULL,
    doctor_id integer NOT NULL,
    diagnosis text,
    notes text,
    attachments jsonb,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.medical_records OWNER TO postgres;

--
-- TOC entry 239 (class 1259 OID 31355)
-- Name: medical_records_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.medical_records_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.medical_records_id_seq OWNER TO postgres;

--
-- TOC entry 5261 (class 0 OID 0)
-- Dependencies: 239
-- Name: medical_records_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.medical_records_id_seq OWNED BY public.medical_records.id;


--
-- TOC entry 242 (class 1259 OID 31367)
-- Name: medicines; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.medicines (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    code character varying(100),
    description text,
    unit character varying(50),
    price integer DEFAULT 0,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.medicines OWNER TO postgres;

--
-- TOC entry 241 (class 1259 OID 31366)
-- Name: medicines_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.medicines_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.medicines_id_seq OWNER TO postgres;

--
-- TOC entry 5262 (class 0 OID 0)
-- Dependencies: 241
-- Name: medicines_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.medicines_id_seq OWNED BY public.medicines.id;


--
-- TOC entry 254 (class 1259 OID 31440)
-- Name: notifications; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.notifications (
    id integer NOT NULL,
    user_id integer,
    type character varying(100),
    payload jsonb,
    is_read boolean DEFAULT false,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.notifications OWNER TO postgres;

--
-- TOC entry 253 (class 1259 OID 31439)
-- Name: notifications_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.notifications_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.notifications_id_seq OWNER TO postgres;

--
-- TOC entry 5263 (class 0 OID 0)
-- Dependencies: 253
-- Name: notifications_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.notifications_id_seq OWNED BY public.notifications.id;


--
-- TOC entry 230 (class 1259 OID 31301)
-- Name: patients; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.patients (
    id integer NOT NULL,
    user_id integer,
    gender character varying(20),
    blood_type character varying(10),
    allergies text,
    emergency_contact jsonb,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    address text DEFAULT ''::text,
    owner_user_id integer,
    full_name character varying(255),
    phone character varying(50),
    email character varying(255),
    dob date,
    occupation character varying(255),
    id_type character varying(50),
    id_number character varying(100),
    nationality character varying(100),
    ethnicity character varying(100),
    old_province character varying(255),
    old_district character varying(255),
    old_ward character varying(255),
    old_street character varying(500),
    new_province character varying(255),
    new_district character varying(255),
    new_ward character varying(255),
    new_street character varying(500),
    id_issue_date date,
    id_issue_place character varying(255),
    zalo boolean DEFAULT false
);


ALTER TABLE public.patients OWNER TO postgres;

--
-- TOC entry 229 (class 1259 OID 31300)
-- Name: patients_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.patients_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.patients_id_seq OWNER TO postgres;

--
-- TOC entry 5264 (class 0 OID 0)
-- Dependencies: 229
-- Name: patients_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.patients_id_seq OWNED BY public.patients.id;


--
-- TOC entry 246 (class 1259 OID 31389)
-- Name: prescriptions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.prescriptions (
    id integer NOT NULL,
    appointment_id integer,
    doctor_id integer NOT NULL,
    patient_id integer NOT NULL,
    items jsonb NOT NULL,
    total_amount integer DEFAULT 0,
    status public.prescription_status DEFAULT 'DRAFT'::public.prescription_status NOT NULL,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    notified_at timestamp with time zone,
    notified_by integer
);


ALTER TABLE public.prescriptions OWNER TO postgres;

--
-- TOC entry 245 (class 1259 OID 31388)
-- Name: prescriptions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.prescriptions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.prescriptions_id_seq OWNER TO postgres;

--
-- TOC entry 5265 (class 0 OID 0)
-- Dependencies: 245
-- Name: prescriptions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.prescriptions_id_seq OWNED BY public.prescriptions.id;


--
-- TOC entry 220 (class 1259 OID 31248)
-- Name: roles; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.roles (
    id integer NOT NULL,
    name character varying(50) NOT NULL,
    description text,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.roles OWNER TO postgres;

--
-- TOC entry 219 (class 1259 OID 31247)
-- Name: roles_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.roles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.roles_id_seq OWNER TO postgres;

--
-- TOC entry 5266 (class 0 OID 0)
-- Dependencies: 219
-- Name: roles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.roles_id_seq OWNED BY public.roles.id;


--
-- TOC entry 232 (class 1259 OID 31312)
-- Name: rooms; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.rooms (
    id integer NOT NULL,
    name character varying(150) NOT NULL,
    type character varying(100),
    description text,
    capacity integer DEFAULT 1,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    doctor_id integer
);


ALTER TABLE public.rooms OWNER TO postgres;

--
-- TOC entry 231 (class 1259 OID 31311)
-- Name: rooms_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.rooms_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.rooms_id_seq OWNER TO postgres;

--
-- TOC entry 5267 (class 0 OID 0)
-- Dependencies: 231
-- Name: rooms_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.rooms_id_seq OWNED BY public.rooms.id;


--
-- TOC entry 234 (class 1259 OID 31323)
-- Name: schedules; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.schedules (
    id integer NOT NULL,
    doctor_id integer NOT NULL,
    room_id integer,
    date date NOT NULL,
    start_time time(6) without time zone NOT NULL,
    end_time time(6) without time zone NOT NULL,
    recurrent_rule character varying(255),
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.schedules OWNER TO postgres;

--
-- TOC entry 233 (class 1259 OID 31322)
-- Name: schedules_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.schedules_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.schedules_id_seq OWNER TO postgres;

--
-- TOC entry 5268 (class 0 OID 0)
-- Dependencies: 233
-- Name: schedules_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.schedules_id_seq OWNED BY public.schedules.id;


--
-- TOC entry 226 (class 1259 OID 31282)
-- Name: specialties; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.specialties (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    slug character varying(255),
    description text,
    image_url text,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.specialties OWNER TO postgres;

--
-- TOC entry 225 (class 1259 OID 31281)
-- Name: specialties_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.specialties_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.specialties_id_seq OWNER TO postgres;

--
-- TOC entry 5269 (class 0 OID 0)
-- Dependencies: 225
-- Name: specialties_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.specialties_id_seq OWNED BY public.specialties.id;


--
-- TOC entry 244 (class 1259 OID 31379)
-- Name: stocks; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stocks (
    id integer NOT NULL,
    medicine_id integer NOT NULL,
    batch_number character varying(100),
    expiry_date date,
    quantity integer DEFAULT 0 NOT NULL,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.stocks OWNER TO postgres;

--
-- TOC entry 243 (class 1259 OID 31378)
-- Name: stocks_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.stocks_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.stocks_id_seq OWNER TO postgres;

--
-- TOC entry 5270 (class 0 OID 0)
-- Dependencies: 243
-- Name: stocks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.stocks_id_seq OWNED BY public.stocks.id;


--
-- TOC entry 248 (class 1259 OID 31402)
-- Name: suppliers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.suppliers (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    contact_info jsonb,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.suppliers OWNER TO postgres;

--
-- TOC entry 247 (class 1259 OID 31401)
-- Name: suppliers_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.suppliers_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.suppliers_id_seq OWNER TO postgres;

--
-- TOC entry 5271 (class 0 OID 0)
-- Dependencies: 247
-- Name: suppliers_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.suppliers_id_seq OWNED BY public.suppliers.id;


--
-- TOC entry 236 (class 1259 OID 31331)
-- Name: timeslots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.timeslots (
    id integer NOT NULL,
    doctor_id integer NOT NULL,
    date date NOT NULL,
    start_time time(6) without time zone NOT NULL,
    end_time time(6) without time zone NOT NULL,
    max_patients integer DEFAULT 1,
    booked_count integer DEFAULT 0,
    is_active boolean DEFAULT true,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.timeslots OWNER TO postgres;

--
-- TOC entry 235 (class 1259 OID 31330)
-- Name: timeslots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.timeslots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.timeslots_id_seq OWNER TO postgres;

--
-- TOC entry 5272 (class 0 OID 0)
-- Dependencies: 235
-- Name: timeslots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.timeslots_id_seq OWNED BY public.timeslots.id;


--
-- TOC entry 222 (class 1259 OID 31258)
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    email character varying(255),
    password character varying(255),
    full_name character varying(255) NOT NULL,
    phone character varying(50),
    dob date,
    avatar_url text,
    is_active boolean DEFAULT true,
    role_id integer NOT NULL,
    created_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp(6) with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.users OWNER TO postgres;

--
-- TOC entry 221 (class 1259 OID 31257)
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO postgres;

--
-- TOC entry 5273 (class 0 OID 0)
-- Dependencies: 221
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- TOC entry 4920 (class 2604 OID 31346)
-- Name: appointments id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.appointments ALTER COLUMN id SET DEFAULT nextval('public.appointments_id_seq'::regclass);


--
-- TOC entry 4958 (class 2604 OID 31454)
-- Name: audit_logs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_logs ALTER COLUMN id SET DEFAULT nextval('public.audit_logs_id_seq'::regclass);


--
-- TOC entry 4960 (class 2604 OID 38296)
-- Name: doctor_reviews id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctor_reviews ALTER COLUMN id SET DEFAULT nextval('public.doctor_reviews_id_seq'::regclass);


--
-- TOC entry 4902 (class 2604 OID 31296)
-- Name: doctor_specialties id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctor_specialties ALTER COLUMN id SET DEFAULT nextval('public.doctor_specialties_id_seq'::regclass);


--
-- TOC entry 4895 (class 2604 OID 31273)
-- Name: doctors id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctors ALTER COLUMN id SET DEFAULT nextval('public.doctors_id_seq'::regclass);


--
-- TOC entry 4943 (class 2604 OID 31415)
-- Name: invoices id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoices ALTER COLUMN id SET DEFAULT nextval('public.invoices_id_seq'::regclass);


--
-- TOC entry 4951 (class 2604 OID 31431)
-- Name: lab_orders id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lab_orders ALTER COLUMN id SET DEFAULT nextval('public.lab_orders_id_seq'::regclass);


--
-- TOC entry 4925 (class 2604 OID 31359)
-- Name: medical_records id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.medical_records ALTER COLUMN id SET DEFAULT nextval('public.medical_records_id_seq'::regclass);


--
-- TOC entry 4928 (class 2604 OID 31370)
-- Name: medicines id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.medicines ALTER COLUMN id SET DEFAULT nextval('public.medicines_id_seq'::regclass);


--
-- TOC entry 4955 (class 2604 OID 31443)
-- Name: notifications id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.notifications ALTER COLUMN id SET DEFAULT nextval('public.notifications_id_seq'::regclass);


--
-- TOC entry 4904 (class 2604 OID 31304)
-- Name: patients id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.patients ALTER COLUMN id SET DEFAULT nextval('public.patients_id_seq'::regclass);


--
-- TOC entry 4936 (class 2604 OID 31392)
-- Name: prescriptions id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prescriptions ALTER COLUMN id SET DEFAULT nextval('public.prescriptions_id_seq'::regclass);


--
-- TOC entry 4889 (class 2604 OID 31251)
-- Name: roles id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.roles ALTER COLUMN id SET DEFAULT nextval('public.roles_id_seq'::regclass);


--
-- TOC entry 4909 (class 2604 OID 31315)
-- Name: rooms id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rooms ALTER COLUMN id SET DEFAULT nextval('public.rooms_id_seq'::regclass);


--
-- TOC entry 4912 (class 2604 OID 31326)
-- Name: schedules id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.schedules ALTER COLUMN id SET DEFAULT nextval('public.schedules_id_seq'::regclass);


--
-- TOC entry 4899 (class 2604 OID 31285)
-- Name: specialties id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.specialties ALTER COLUMN id SET DEFAULT nextval('public.specialties_id_seq'::regclass);


--
-- TOC entry 4932 (class 2604 OID 31382)
-- Name: stocks id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stocks ALTER COLUMN id SET DEFAULT nextval('public.stocks_id_seq'::regclass);


--
-- TOC entry 4941 (class 2604 OID 31405)
-- Name: suppliers id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.suppliers ALTER COLUMN id SET DEFAULT nextval('public.suppliers_id_seq'::regclass);


--
-- TOC entry 4914 (class 2604 OID 31334)
-- Name: timeslots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.timeslots ALTER COLUMN id SET DEFAULT nextval('public.timeslots_id_seq'::regclass);


--
-- TOC entry 4891 (class 2604 OID 31261)
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- TOC entry 5205 (class 0 OID 31207)
-- Dependencies: 218
-- Data for Name: _prisma_migrations; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public._prisma_migrations (id, checksum, finished_at, migration_name, logs, rolled_back_at, started_at, applied_steps_count) FROM stdin;
83c06f1b-3e2c-495b-81eb-50b71b085c93	f48d736a857abd469a5b1d2241fe999eb3492f2faafb2bbe41986ede019b6026	2025-11-16 16:32:02.03209+07	20251114173212_add_specialties	\N	\N	2025-11-16 16:32:01.931078+07	1
af2eb269-9b62-400c-87be-359cb56fa1d0	d04852525fd017d784f9662bacaa8261d9ebb2a4b16aee591de00193065a3f03	2025-11-18 09:43:21.823217+07	20251118_add_patient_profile_fields	\N	\N	2025-11-18 09:43:21.803905+07	1
462e1fa5-ba10-4ebf-af2e-9928baa7a5ea	b67c70c806b69318c9cfde9a352374dff816a5514df43fe8169310f8a481b017	2025-11-16 16:32:03.412615+07	20251116093203_add_room_doctorid	\N	\N	2025-11-16 16:32:03.409137+07	1
31cbee38-ea0b-4b67-ab49-34104bd02a4d	731b2f001fc055d236f42b05a332eb205c7b2e2176c9ba1df9688a7f14fb11d8	\N	20251118_add_doctor_reviews	A migration failed to apply. New migrations cannot be applied before the error is recovered from. Read more about how to resolve migration issues in a production database: https://pris.ly/d/migrate-resolve\n\nMigration name: 20251118_add_doctor_reviews\n\nDatabase error code: 42P07\n\nDatabase error:\nERROR: relation "doctor_reviews" already exists\n\nDbError { severity: "ERROR", parsed_severity: Some(Error), code: SqlState(E42P07), message: "relation \\"doctor_reviews\\" already exists", detail: None, hint: None, position: None, where_: None, schema: None, table: None, column: None, datatype: None, constraint: None, file: Some("heap.c"), line: Some(1162), routine: Some("heap_create_with_catalog") }\n\n   0: sql_schema_connector::apply_migration::apply_script\n           with migration_name="20251118_add_doctor_reviews"\n             at schema-engine\\connectors\\sql-schema-connector\\src\\apply_migration.rs:106\n   1: schema_core::commands::apply_migrations::Applying migration\n           with migration_name="20251118_add_doctor_reviews"\n             at schema-engine\\core\\src\\commands\\apply_migrations.rs:91\n   2: schema_core::state::ApplyMigrations\n             at schema-engine\\core\\src\\state.rs:226	2025-11-18 01:23:48.615812+07	2025-11-18 01:23:43.509151+07	0
07a1aecd-0694-461a-9327-f8279e805940	95e4f1e56352aac64ea09153cd63672caed4c68e624d1d8e53cd1dfd5a0a9c3a	2025-11-16 19:26:39.283727+07	20251116122639_add_doctor_fields	\N	\N	2025-11-16 19:26:39.277961+07	1
3af19181-786b-48c2-be11-bf7fe5282173	731b2f001fc055d236f42b05a332eb205c7b2e2176c9ba1df9688a7f14fb11d8	2025-11-18 01:23:48.617712+07	20251118_add_doctor_reviews		\N	2025-11-18 01:23:48.617712+07	0
579574b3-8326-485d-9d7b-686288f48154	61316cb37a9fb24e6d57426e9dfccbc64700a54b88d19d054150921543e2cc9c	\N	20251116_add_address_to_patients	A migration failed to apply. New migrations cannot be applied before the error is recovered from. Read more about how to resolve migration issues in a production database: https://pris.ly/d/migrate-resolve\n\nMigration name: 20251116_add_address_to_patients\n\nDatabase error code: 42701\n\nDatabase error:\nERROR: column "address" of relation "patients" already exists\n\nDbError { severity: "ERROR", parsed_severity: Some(Error), code: SqlState(E42701), message: "column \\"address\\" of relation \\"patients\\" already exists", detail: None, hint: None, position: None, where_: None, schema: None, table: None, column: None, datatype: None, constraint: None, file: Some("tablecmds.c"), line: Some(7481), routine: Some("check_for_column_name_collision") }\n\n   0: sql_schema_connector::apply_migration::apply_script\n           with migration_name="20251116_add_address_to_patients"\n             at schema-engine\\connectors\\sql-schema-connector\\src\\apply_migration.rs:106\n   1: schema_core::commands::apply_migrations::Applying migration\n           with migration_name="20251116_add_address_to_patients"\n             at schema-engine\\core\\src\\commands\\apply_migrations.rs:91\n   2: schema_core::state::ApplyMigrations\n             at schema-engine\\core\\src\\state.rs:226	2025-11-18 01:23:15.16992+07	2025-11-18 01:23:05.104359+07	0
364341bb-fe47-4955-a81c-1ded42e7ddc4	7ceaaed63f6f6faebe824610090430f73e5a29ad495f828629c536500326fd28	2025-11-18 08:51:40.741435+07	20251118_add_patient_owner_and_fields		\N	2025-11-18 08:51:40.741435+07	0
e1685a57-4a3e-474f-8eb5-ad1d74270ac5	61316cb37a9fb24e6d57426e9dfccbc64700a54b88d19d054150921543e2cc9c	\N	20251116_add_address_to_patients	A migration failed to apply. New migrations cannot be applied before the error is recovered from. Read more about how to resolve migration issues in a production database: https://pris.ly/d/migrate-resolve\n\nMigration name: 20251116_add_address_to_patients\n\nDatabase error code: 42701\n\nDatabase error:\nERROR: column "address" of relation "patients" already exists\n\nDbError { severity: "ERROR", parsed_severity: Some(Error), code: SqlState(E42701), message: "column \\"address\\" of relation \\"patients\\" already exists", detail: None, hint: None, position: None, where_: None, schema: None, table: None, column: None, datatype: None, constraint: None, file: Some("tablecmds.c"), line: Some(7481), routine: Some("check_for_column_name_collision") }\n\n   0: sql_schema_connector::apply_migration::apply_script\n           with migration_name="20251116_add_address_to_patients"\n             at schema-engine\\connectors\\sql-schema-connector\\src\\apply_migration.rs:106\n   1: schema_core::commands::apply_migrations::Applying migration\n           with migration_name="20251116_add_address_to_patients"\n             at schema-engine\\core\\src\\commands\\apply_migrations.rs:91\n   2: schema_core::state::ApplyMigrations\n             at schema-engine\\core\\src\\state.rs:226	2025-11-18 01:23:38.920176+07	2025-11-18 01:23:24.625615+07	0
005b0807-985f-4c54-9e2b-0a556520ddb3	61316cb37a9fb24e6d57426e9dfccbc64700a54b88d19d054150921543e2cc9c	2025-11-18 01:23:38.924223+07	20251116_add_address_to_patients		\N	2025-11-18 01:23:38.924223+07	0
be9a1792-4f9b-4723-bed5-00c871f62288	09f52240b52ad909c314c9fa3db01462dbe3a057327e1dbb7f3faa5acddace24	2025-11-21 17:47:00.412081+07	20251120_add_prescription_id	\N	\N	2025-11-21 17:47:00.360677+07	1
c5b87cad-d000-42e5-a8b9-23afca5dad8f	ff2a3b947e56da26fb84462e01c255b0b46e0bd6b80d6631f437c08cc5beb0d2	\N	20251121_add_timeslot_unique_constraint	A migration failed to apply. New migrations cannot be applied before the error is recovered from. Read more about how to resolve migration issues in a production database: https://pris.ly/d/migrate-resolve\n\nMigration name: 20251121_add_timeslot_unique_constraint\n\nDatabase error code: 42P07\n\nDatabase error:\nERROR: relation "timeslot_unique" already exists\n\nDbError { severity: "ERROR", parsed_severity: Some(Error), code: SqlState(E42P07), message: "relation \\"timeslot_unique\\" already exists", detail: None, hint: None, position: None, where_: None, schema: None, table: None, column: None, datatype: None, constraint: None, file: Some("index.c"), line: Some(900), routine: Some("index_create") }\n\n   0: sql_schema_connector::apply_migration::apply_script\n           with migration_name="20251121_add_timeslot_unique_constraint"\n             at schema-engine\\connectors\\sql-schema-connector\\src\\apply_migration.rs:106\n   1: schema_core::commands::apply_migrations::Applying migration\n           with migration_name="20251121_add_timeslot_unique_constraint"\n             at schema-engine\\core\\src\\commands\\apply_migrations.rs:91\n   2: schema_core::state::ApplyMigrations\n             at schema-engine\\core\\src\\state.rs:226	2025-11-21 17:47:15.116363+07	2025-11-21 17:47:00.421744+07	0
b1ae5b79-3517-4f12-9f68-6556edec64a2	1bd512d58a60db2b1b7a37b37d101aae5f654c9c598e06a8ef592e104d484444	2025-11-21 17:47:22.028913+07	20251121_add_timeslot_unique_constraint	\N	\N	2025-11-21 17:47:22.00726+07	1
\.


--
-- TOC entry 5225 (class 0 OID 31343)
-- Dependencies: 238
-- Data for Name: appointments; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.appointments (id, patient_id, doctor_id, timeslot_id, appointment_date, appointment_time, status, reason, source, created_by, created_at, updated_at) FROM stdin;
1	1	1	\N	2025-11-15	01:00:00	COMPLETED	General checkup	web	9	2025-11-16 17:09:31.773+07	2025-11-16 17:09:31.773+07
4	4	1	\N	2025-11-15	04:00:00	COMPLETED	General checkup	web	12	2025-11-16 17:09:31.778+07	2025-11-16 17:09:31.778+07
2	2	1	\N	2025-11-15	02:00:00	NO_SHOW	General checkup	web	10	2025-11-16 17:09:31.776+07	2025-11-16 18:00:00.596+07
3	3	1	\N	2025-11-15	03:00:00	NO_SHOW	General checkup	web	11	2025-11-16 17:09:31.777+07	2025-11-16 18:00:00.596+07
5	5	5	\N	2025-11-17	08:00:00	NO_SHOW	đau đầu	online	4	2025-11-16 18:55:40.13+07	2025-11-18 09:00:00.761+07
6	5	5	\N	2025-11-17	08:40:00	NO_SHOW	aaaa	online	4	2025-11-16 20:30:08.576+07	2025-11-18 09:00:00.761+07
11	10	5	\N	2025-11-17	08:20:00	NO_SHOW	áđâsđâsđâsdsad	web	3	2025-11-17 21:23:42.998+07	2025-11-18 09:00:00.761+07
15	5	5	\N	2025-11-18	08:20:00	NO_SHOW	sdaaaaaaaaâd	online	4	2025-11-18 08:02:02.702+07	2025-11-19 17:00:00.435+07
16	5	5	\N	2025-11-18	08:40:00	NO_SHOW	áddddddddđ	online	4	2025-11-18 08:05:17.12+07	2025-11-19 17:00:00.435+07
18	14	5	\N	2025-11-18	09:00:00	NO_SHOW	aaaaaaaâ	online	4	2025-11-18 14:09:13.807+07	2025-11-19 17:00:00.435+07
19	15	5	\N	2025-11-18	09:20:00	NO_SHOW	aaaaaaaaaaaâ	online	4	2025-11-18 14:27:06.192+07	2025-11-19 17:00:00.435+07
13	5	5	\N	2025-11-18	08:00:00	NO_SHOW	sđâsđâsdsađâsdấđá	online	4	2025-11-18 07:49:47.306+07	2025-11-19 17:00:00.435+07
14	5	5	\N	2025-11-18	10:40:00	NO_SHOW	ádâsdấđâsđâsđâsdsad	online	4	2025-11-18 07:57:22.221+07	2025-11-19 17:00:00.435+07
29	15	5	651	2025-11-22	08:00:00	PENDING	áđâsád	online	4	2025-11-21 19:26:40.054+07	2025-11-21 19:26:40.054+07
28	15	5	993	2025-11-26	08:00:00	CHECKED_IN	bệnh	online	4	2025-11-21 18:52:13.747+07	2025-11-21 21:37:03.844+07
26	15	1	52	2025-11-22	11:00:00	CANCELLED	áđâsdá\nCancellation reason: Hủy lịch hẹn	online	4	2025-11-21 15:40:27.595+07	2025-11-21 21:37:13.384+07
27	15	1	50	2025-11-22	09:00:00	CANCELLED	aaâ\nCancellation reason: Xóa lịch hẹn	online	4	2025-11-21 16:07:39.708+07	2025-11-21 21:37:16.706+07
25	15	1	49	2025-11-22	08:00:00	CANCELLED	hi\nCancellation reason: Xóa lịch hẹn\nCancellation reason: Xóa lịch hẹn\nCancellation reason: Xóa lịch hẹn	online	4	2025-11-21 15:36:00.281+07	2025-11-21 21:37:32.55+07
51	15	5	652	2025-11-22	08:30:00	PENDING	áđâsđâsd	online	4	2025-11-21 22:15:48.527+07	2025-11-21 22:15:48.527+07
52	15	5	653	2025-11-22	09:00:00	PENDING	áđá	online	4	2025-11-21 22:52:50.823+07	2025-11-21 22:52:50.823+07
53	15	5	654	2025-11-22	09:30:00	PENDING	áđâsd	online	4	2025-11-21 22:53:10.289+07	2025-11-21 22:53:10.289+07
54	15	5	655	2025-11-22	10:00:00	PENDING	bệnh	online	4	2025-11-21 23:23:46.971+07	2025-11-21 23:23:46.971+07
55	14	5	656	2025-11-22	10:30:00	PENDING	áđâsđá	online	4	2025-11-21 23:49:21.669+07	2025-11-21 23:49:21.669+07
7	6	1	\N	2025-11-16	08:00:00	COMPLETED	General checkup	web	22	2025-11-17 19:42:42.592+07	2025-11-17 19:42:42.592+07
30	1	1	\N	2025-11-26	08:00:00	CHECKED_IN	Khám sức khỏe định kỳ 1	web	\N	2025-11-21 19:31:09.759+07	2025-11-21 23:24:50.764+07
8	7	1	\N	2025-11-16	09:00:00	NO_SHOW	General checkup	web	23	2025-11-17 19:42:42.617+07	2025-11-17 20:00:00.58+07
9	8	1	\N	2025-11-16	10:00:00	NO_SHOW	General checkup	web	24	2025-11-17 19:42:42.633+07	2025-11-17 20:00:00.58+07
10	9	1	\N	2025-11-16	11:00:00	COMPLETED	General checkup	web	25	2025-11-17 19:42:42.654+07	2025-11-17 19:42:42.654+07
17	5	3	\N	2025-11-18	08:00:00	NO_SHOW	ádddddddddddddddddddddđ	online	4	2025-11-18 08:05:29.629+07	2025-11-19 17:00:00.435+07
12	5	5	\N	2025-11-18	08:00:00	NO_SHOW	áđaádâda	online	4	2025-11-17 21:54:35.546+07	2025-11-19 17:00:00.435+07
23	16	5	\N	2025-11-18	16:40:00	NO_SHOW	aaaaaaaaaaa	web	3	2025-11-18 16:29:48.49+07	2025-11-19 17:00:00.435+07
20	15	5	\N	2025-11-20	08:00:00	NO_SHOW	aaabádưpábhăbh bdábhdhsa djhưqbhj ebjqhwqe	online	4	2025-11-18 15:00:52.704+07	2025-11-21 16:00:00.53+07
21	16	5	\N	2025-11-20	08:30:00	NO_SHOW	aaaaaaaaaa	web	3	2025-11-18 16:04:41.758+07	2025-11-21 16:00:00.53+07
22	16	5	\N	2025-11-20	09:00:00	NO_SHOW	aaaaaaaaaaâ	web	3	2025-11-18 16:15:15.889+07	2025-11-21 16:00:00.53+07
24	26	5	\N	2025-11-20	09:30:00	NO_SHOW	áddddddddd	web	3	2025-11-18 17:07:20.916+07	2025-11-21 16:00:00.53+07
\.


--
-- TOC entry 5243 (class 0 OID 31451)
-- Dependencies: 256
-- Data for Name: audit_logs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.audit_logs (id, user_id, action, meta, created_at) FROM stdin;
1	1	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.814+07
2	2	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.817+07
3	3	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.818+07
4	4	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.819+07
5	5	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.82+07
6	6	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.82+07
7	7	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.821+07
8	8	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.822+07
9	9	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.823+07
10	10	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.824+07
11	11	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.826+07
12	12	seed:created	{"source": "seed_fresh.js"}	2025-11-16 17:09:31.827+07
13	14	seed:created	{"source": "seed_all.js", "timestamp": "2025-11-17T12:42:43.405Z"}	2025-11-17 19:42:43.407+07
14	15	seed:created	{"source": "seed_all.js", "timestamp": "2025-11-17T12:42:43.419Z"}	2025-11-17 19:42:43.421+07
15	16	seed:created	{"source": "seed_all.js", "timestamp": "2025-11-17T12:42:43.427Z"}	2025-11-17 19:42:43.429+07
16	17	seed:created	{"source": "seed_all.js", "timestamp": "2025-11-17T12:42:43.438Z"}	2025-11-17 19:42:43.441+07
17	18	seed:created	{"source": "seed_all.js", "timestamp": "2025-11-17T12:42:43.452Z"}	2025-11-17 19:42:43.455+07
18	19	seed:created	{"source": "seed_all.js", "timestamp": "2025-11-17T12:42:43.464Z"}	2025-11-17 19:42:43.466+07
19	20	seed:created	{"source": "seed_all.js", "timestamp": "2025-11-17T12:42:43.474Z"}	2025-11-17 19:42:43.477+07
20	21	seed:created	{"source": "seed_all.js", "timestamp": "2025-11-17T12:42:43.484Z"}	2025-11-17 19:42:43.486+07
\.


--
-- TOC entry 5245 (class 0 OID 38293)
-- Dependencies: 258
-- Data for Name: doctor_reviews; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.doctor_reviews (id, doctor_id, patient_id, rating, comment, created_at, updated_at) FROM stdin;
\.


--
-- TOC entry 5215 (class 0 OID 31293)
-- Dependencies: 228
-- Data for Name: doctor_specialties; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.doctor_specialties (id, doctor_id, specialty_id, created_at) FROM stdin;
1	1	1	2025-11-16 17:09:31.571+07
2	2	2	2025-11-16 17:09:31.574+07
3	3	3	2025-11-16 17:09:31.574+07
4	4	4	2025-11-16 17:09:31.575+07
5	6	1	2025-11-17 19:42:39.126+07
6	7	2	2025-11-17 19:42:39.166+07
7	8	3	2025-11-17 19:42:39.195+07
8	9	4	2025-11-17 19:42:39.228+07
\.


--
-- TOC entry 5211 (class 0 OID 31270)
-- Dependencies: 224
-- Data for Name: doctors; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.doctors (id, user_id, license_number, specialties, bio, consultation_fee, rating, created_at, updated_at, address, gender) FROM stdin;
1	5	LIC-5	{Cardiology}	\N	200000	4.50	2025-11-16 17:09:31.154+07	2025-11-16 17:09:31.154+07	\N	\N
2	6	LIC-6	{Dermatology}	\N	200000	4.50	2025-11-16 17:09:31.215+07	2025-11-16 17:09:31.215+07	\N	\N
4	8	LIC-8	{Pediatrics}	\N	200000	4.50	2025-11-16 17:09:31.332+07	2025-11-16 17:09:31.332+07	\N	\N
3	7	LIC-7	{Dermatology}	\N	200000	4.50	2025-11-16 17:09:31.274+07	2025-11-16 17:09:31.274+07	\N	\N
5	2	LIC-1	{General}	qưeqưeqưeqưe	200000	4.50	2025-11-16 17:09:31.154+07	2025-11-16 17:09:31.154+07	123123	MALE
6	18	LIC-18	\N	\N	200000	4.50	2025-11-17 19:42:36.696+07	2025-11-17 19:42:36.696+07	\N	\N
7	19	LIC-19	\N	\N	200000	4.50	2025-11-17 19:42:37.076+07	2025-11-17 19:42:37.076+07	\N	\N
8	20	LIC-20	\N	\N	200000	4.50	2025-11-17 19:42:37.408+07	2025-11-17 19:42:37.408+07	\N	\N
9	21	LIC-21	\N	\N	200000	4.50	2025-11-17 19:42:37.786+07	2025-11-17 19:42:37.786+07	\N	\N
\.


--
-- TOC entry 5237 (class 0 OID 31412)
-- Dependencies: 250
-- Data for Name: invoices; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.invoices (id, appointment_id, patient_id, items, subtotal, tax, discount, total, status, paid_at, created_at, updated_at, prescription_id) FROM stdin;
1	1	1	[{"desc": "Consultation", "amount": 200000}, {"desc": "Lab", "amount": 150000}]	350000	35000	0	385000	PAID	2025-11-16 17:09:31.798+07	2025-11-16 17:09:31.799+07	2025-11-16 17:09:31.799+07	\N
2	2	2	[{"desc": "Consultation", "amount": 200000}, {"desc": "Lab", "amount": 150000}]	350000	35000	0	385000	UNPAID	\N	2025-11-16 17:09:31.803+07	2025-11-16 17:09:31.803+07	\N
3	3	3	[{"desc": "Consultation", "amount": 200000}, {"desc": "Lab", "amount": 150000}]	350000	35000	0	385000	UNPAID	\N	2025-11-16 17:09:31.806+07	2025-11-16 17:09:31.806+07	\N
4	4	4	[{"desc": "Consultation", "amount": 200000}, {"desc": "Lab", "amount": 150000}]	350000	35000	0	385000	PAID	2025-11-16 17:09:31.808+07	2025-11-16 17:09:31.809+07	2025-11-16 17:09:31.809+07	\N
5	5	5	[{"desc": "Consultation", "amount": 200000}, {"desc": "Lab tests", "amount": 150000}, {"desc": "Medicines", "amount": 250000}]	350000	35000	0	660000	UNPAID	\N	2025-11-17 19:42:43.137+07	2025-11-17 19:42:43.137+07	\N
6	6	5	[{"desc": "Consultation", "amount": 200000}, {"desc": "Lab tests", "amount": 150000}, {"desc": "Medicines", "amount": 250000}]	350000	35000	0	660000	UNPAID	\N	2025-11-17 19:42:43.15+07	2025-11-17 19:42:43.15+07	\N
7	7	6	[{"desc": "Consultation", "amount": 200000}, {"desc": "Lab tests", "amount": 150000}, {"desc": "Medicines", "amount": 250000}]	350000	35000	0	660000	PAID	2025-11-17 19:42:43.158+07	2025-11-17 19:42:43.16+07	2025-11-17 19:42:43.16+07	\N
8	8	7	[{"desc": "Consultation", "amount": 200000}, {"desc": "Lab tests", "amount": 150000}, {"desc": "Medicines", "amount": 250000}]	350000	35000	0	660000	UNPAID	\N	2025-11-17 19:42:43.17+07	2025-11-17 19:42:43.17+07	\N
9	9	8	[{"desc": "Consultation", "amount": 200000}, {"desc": "Lab tests", "amount": 150000}, {"desc": "Medicines", "amount": 250000}]	350000	35000	0	660000	UNPAID	\N	2025-11-17 19:42:43.181+07	2025-11-17 19:42:43.181+07	\N
10	10	9	[{"desc": "Consultation", "amount": 200000}, {"desc": "Lab tests", "amount": 150000}, {"desc": "Medicines", "amount": 250000}]	350000	35000	0	660000	PAID	2025-11-17 19:42:43.189+07	2025-11-17 19:42:43.191+07	2025-11-17 19:42:43.191+07	\N
11	\N	26	[{"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	24000	0	0	24000	UNPAID	\N	2025-11-20 00:41:58.681+07	2025-11-20 00:41:58.681+07	19
12	\N	26	[{"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	24000	0	0	24000	UNPAID	\N	2025-11-20 00:50:28.543+07	2025-11-20 00:50:28.543+07	19
13	\N	26	[{"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	24000	0	0	24000	UNPAID	\N	2025-11-20 00:51:20.208+07	2025-11-20 00:51:20.208+07	19
14	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	224000	0	0	224000	UNPAID	\N	2025-11-20 01:17:15.281+07	2025-11-20 01:17:15.281+07	\N
15	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	224000	0	0	224000	UNPAID	\N	2025-11-20 01:21:21.747+07	2025-11-20 01:21:21.747+07	\N
16	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	224000	0	0	224000	UNPAID	\N	2025-11-20 01:47:46.603+07	2025-11-20 01:47:46.603+07	\N
17	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	224000	0	0	224000	UNPAID	\N	2025-11-20 01:48:10.122+07	2025-11-20 01:48:10.122+07	\N
18	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	224000	0	0	224000	UNPAID	\N	2025-11-20 01:50:07.432+07	2025-11-20 01:50:07.432+07	\N
19	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	224000	0	0	224000	UNPAID	\N	2025-11-20 02:35:03.164+07	2025-11-20 02:35:03.164+07	\N
20	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	224000	0	0	224000	UNPAID	\N	2025-11-20 02:35:17.85+07	2025-11-20 02:35:17.85+07	\N
21	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	224000	0	0	224000	UNPAID	\N	2025-11-20 02:37:32.253+07	2025-11-20 02:37:32.253+07	\N
22	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	224000	0	0	224000	UNPAID	\N	2025-11-20 02:37:54.69+07	2025-11-20 02:37:54.69+07	\N
24	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 48000, "quantity": 4, "unit_price": 12000, "description": "Aspirin"}]	248000	0	0	248000	PAID	2025-11-20 02:44:54.216+07	2025-11-20 02:39:33.389+07	2025-11-20 02:44:54.216+07	\N
23	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 24000, "quantity": 2, "unit_price": 12000, "description": "Aspirin"}]	224000	0	0	224000	PAID	2025-11-20 03:25:56.305+07	2025-11-20 02:39:19.349+07	2025-11-20 03:25:56.305+07	\N
25	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 150000, "quantity": 6, "unit_price": 25000, "description": "Amoxicillin"}]	350000	0	0	350000	PAID	2025-11-20 05:12:28.432+07	2025-11-20 05:12:22.408+07	2025-11-20 05:12:28.432+07	24
26	\N	26	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 200000, "quantity": 20, "unit_price": 10000, "description": "Paracetamol"}]	400000	0	0	400000	PAID	2025-11-20 05:55:04.073+07	2025-11-20 05:54:58.09+07	2025-11-20 05:55:04.073+07	25
27	\N	1	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 600000, "quantity": 50, "unit_price": 12000, "description": "Aspirin"}, {"type": "medicine", "amount": 300000, "quantity": 30, "unit_price": 10000, "description": "Paracetamol"}]	1100000	0	0	1100000	PAID	2025-11-20 06:06:23.495+07	2025-11-20 06:06:11.855+07	2025-11-20 06:06:23.495+07	26
28	\N	27	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 20000, "quantity": 2, "unit_price": 10000, "description": "Paracetamol"}]	220000	0	0	220000	PAID	2025-11-21 15:44:02.744+07	2025-11-21 15:43:47.259+07	2025-11-21 15:44:02.744+07	27
29	\N	15	[{"type": "consultation", "amount": 200000, "quantity": 1, "unit_price": 200000, "description": "Phí khám"}, {"type": "medicine", "amount": 20000, "quantity": 2, "unit_price": 10000, "description": "Paracetamol"}]	220000	0	0	220000	PAID	2025-11-21 23:28:49.604+07	2025-11-21 23:28:33.983+07	2025-11-21 23:28:49.604+07	28
\.


--
-- TOC entry 5239 (class 0 OID 31428)
-- Dependencies: 252
-- Data for Name: lab_orders; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.lab_orders (id, appointment_id, patient_id, doctor_id, tests, status, results, created_at, updated_at) FROM stdin;
1	1	1	1	[{"name": "CBC"}, {"name": "Blood Type"}]	COMPLETED	{"CBC": "Normal"}	2025-11-16 17:09:31.796+07	2025-11-16 17:09:31.796+07
2	2	2	1	[{"name": "CBC"}, {"name": "Blood Type"}]	COMPLETED	{"CBC": "Normal"}	2025-11-16 17:09:31.803+07	2025-11-16 17:09:31.803+07
3	3	3	1	[{"name": "CBC"}, {"name": "Blood Type"}]	COMPLETED	{"CBC": "Normal"}	2025-11-16 17:09:31.805+07	2025-11-16 17:09:31.805+07
4	4	4	1	[{"name": "CBC"}, {"name": "Blood Type"}]	COMPLETED	{"CBC": "Normal"}	2025-11-16 17:09:31.808+07	2025-11-16 17:09:31.808+07
5	5	5	5	[{"name": "CBC", "status": "completed"}, {"name": "Blood Type", "status": "completed"}]	COMPLETED	{"CBC": "Normal", "BloodType": "O+"}	2025-11-17 19:42:43.035+07	2025-11-17 19:42:43.035+07
6	6	5	5	[{"name": "CBC", "status": "completed"}, {"name": "Blood Type", "status": "completed"}]	COMPLETED	{"CBC": "Normal", "BloodType": "O+"}	2025-11-17 19:42:43.059+07	2025-11-17 19:42:43.059+07
7	7	6	1	[{"name": "CBC", "status": "completed"}, {"name": "Blood Type", "status": "completed"}]	COMPLETED	{"CBC": "Normal", "BloodType": "O+"}	2025-11-17 19:42:43.071+07	2025-11-17 19:42:43.071+07
8	8	7	1	[{"name": "CBC", "status": "completed"}, {"name": "Blood Type", "status": "completed"}]	COMPLETED	{"CBC": "Normal", "BloodType": "O+"}	2025-11-17 19:42:43.079+07	2025-11-17 19:42:43.079+07
9	9	8	1	[{"name": "CBC", "status": "completed"}, {"name": "Blood Type", "status": "completed"}]	COMPLETED	{"CBC": "Normal", "BloodType": "O+"}	2025-11-17 19:42:43.088+07	2025-11-17 19:42:43.088+07
10	10	9	1	[{"name": "CBC", "status": "completed"}, {"name": "Blood Type", "status": "completed"}]	COMPLETED	{"CBC": "Normal", "BloodType": "O+"}	2025-11-17 19:42:43.098+07	2025-11-17 19:42:43.098+07
\.


--
-- TOC entry 5227 (class 0 OID 31356)
-- Dependencies: 240
-- Data for Name: medical_records; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.medical_records (id, appointment_id, patient_id, doctor_id, diagnosis, notes, attachments, created_at, updated_at) FROM stdin;
1	1	1	1	Good health condition	Routine checkup	\N	2025-11-16 17:09:31.791+07	2025-11-16 17:09:31.791+07
2	2	2	1	Good health condition	Routine checkup	\N	2025-11-16 17:09:31.801+07	2025-11-16 17:09:31.801+07
3	3	3	1	Good health condition	Routine checkup	\N	2025-11-16 17:09:31.804+07	2025-11-16 17:09:31.804+07
4	4	4	1	Good health condition	Routine checkup	\N	2025-11-16 17:09:31.807+07	2025-11-16 17:09:31.807+07
5	5	5	5	Good health condition	Routine checkup. All vitals normal.	\N	2025-11-17 19:42:42.797+07	2025-11-17 19:42:42.797+07
6	6	5	5	Good health condition	Routine checkup. All vitals normal.	\N	2025-11-17 19:42:42.814+07	2025-11-17 19:42:42.814+07
7	7	6	1	Good health condition	Routine checkup. All vitals normal.	\N	2025-11-17 19:42:42.827+07	2025-11-17 19:42:42.827+07
8	8	7	1	Good health condition	Routine checkup. All vitals normal.	\N	2025-11-17 19:42:42.841+07	2025-11-17 19:42:42.841+07
9	9	8	1	Good health condition	Routine checkup. All vitals normal.	\N	2025-11-17 19:42:42.856+07	2025-11-17 19:42:42.856+07
10	10	9	1	Good health condition	Routine checkup. All vitals normal.	\N	2025-11-17 19:42:42.866+07	2025-11-17 19:42:42.866+07
\.


--
-- TOC entry 5229 (class 0 OID 31367)
-- Dependencies: 242
-- Data for Name: medicines; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.medicines (id, name, code, description, unit, price, created_at, updated_at) FROM stdin;
1	Paracetamol	PARA-100	\N	tablet	10000	2025-11-16 17:09:31.78+07	2025-11-16 17:09:31.78+07
2	Ibuprofen	IBU-200	\N	tablet	15000	2025-11-16 17:09:31.784+07	2025-11-16 17:09:31.784+07
3	Amoxicillin	AMOX-500	\N	tablet	25000	2025-11-16 17:09:31.785+07	2025-11-16 17:09:31.785+07
4	Aspirin	ASP-81	\N	tablet	12000	2025-11-16 17:09:31.786+07	2025-11-16 17:09:31.786+07
\.


--
-- TOC entry 5241 (class 0 OID 31440)
-- Dependencies: 254
-- Data for Name: notifications; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.notifications (id, user_id, type, payload, is_read, created_at) FROM stdin;
1	1	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.811+07
2	2	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.816+07
4	4	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.818+07
5	5	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.819+07
6	6	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.82+07
7	7	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.821+07
8	8	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.822+07
9	9	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.823+07
10	10	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.823+07
11	11	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.825+07
12	12	welcome	{"message": "Welcome!"}	f	2025-11-16 17:09:31.826+07
13	14	welcome	{"seeded": true, "message": "Welcome to Clinic Management System"}	f	2025-11-17 19:42:43.273+07
14	15	welcome	{"seeded": true, "message": "Welcome to Clinic Management System"}	f	2025-11-17 19:42:43.287+07
15	16	welcome	{"seeded": true, "message": "Welcome to Clinic Management System"}	f	2025-11-17 19:42:43.295+07
16	17	welcome	{"seeded": true, "message": "Welcome to Clinic Management System"}	f	2025-11-17 19:42:43.308+07
17	18	welcome	{"seeded": true, "message": "Welcome to Clinic Management System"}	f	2025-11-17 19:42:43.315+07
18	19	welcome	{"seeded": true, "message": "Welcome to Clinic Management System"}	f	2025-11-17 19:42:43.325+07
19	20	welcome	{"seeded": true, "message": "Welcome to Clinic Management System"}	f	2025-11-17 19:42:43.333+07
20	21	welcome	{"seeded": true, "message": "Welcome to Clinic Management System"}	f	2025-11-17 19:42:43.341+07
21	4	APPOINTMENT_REMINDER	{"date": "2025-11-18T00:00:00.000Z", "time": "1970-01-01T08:00:00.000Z", "message": "Nhắc nhở: Bạn có lịch hẹn vào ngày mai", "doctor_name": "Doctor Account", "appointment_id": 12}	f	2025-11-18 08:01:03.804+07
22	4	APPOINTMENT_REMINDER	{"date": "2025-11-18T00:00:00.000Z", "time": "1970-01-01T08:00:00.000Z", "message": "Nhắc nhở: Bạn có lịch hẹn vào ngày mai", "doctor_name": "Doctor Account", "appointment_id": 13}	f	2025-11-18 08:01:03.804+07
23	4	APPOINTMENT_REMINDER	{"date": "2025-11-18T00:00:00.000Z", "time": "1970-01-01T10:40:00.000Z", "message": "Nhắc nhở: Bạn có lịch hẹn vào ngày mai", "doctor_name": "Doctor Account", "appointment_id": 14}	f	2025-11-18 08:01:03.804+07
3	3	welcome	{"message": "Welcome!"}	t	2025-11-16 17:09:31.817+07
24	32	prescription_ready	{"message": "Prescription #25 is ready for dispensing (Invoice #26).", "invoice_id": 26, "patient_id": 26, "patient_name": "àqưeãzc", "patient_phone": "0987654321", "patient_user_id": 30, "prescription_id": 25}	f	2025-11-20 05:55:04.089+07
25	32	prescription_ready	{"message": "Prescription #26 is ready for dispensing (Invoice #27).", "invoice_id": 27, "patient_id": 1, "patient_name": "Patient One", "patient_phone": null, "patient_user_id": 9, "prescription_id": 26}	f	2025-11-20 06:06:23.511+07
26	\N	APPOINTMENT_REMINDER	{"date": "2025-11-20T00:00:00.000Z", "time": "1970-01-01T08:00:00.000Z", "message": "Nhắc nhở: Bạn có lịch hẹn vào ngày mai", "doctor_name": "Doctor Account", "appointment_id": 20}	f	2025-11-20 08:00:10.618+07
27	27	APPOINTMENT_REMINDER	{"date": "2025-11-20T00:00:00.000Z", "time": "1970-01-01T08:30:00.000Z", "message": "Nhắc nhở: Bạn có lịch hẹn vào ngày mai", "doctor_name": "Doctor Account", "appointment_id": 21}	f	2025-11-20 08:00:10.618+07
28	27	APPOINTMENT_REMINDER	{"date": "2025-11-20T00:00:00.000Z", "time": "1970-01-01T09:00:00.000Z", "message": "Nhắc nhở: Bạn có lịch hẹn vào ngày mai", "doctor_name": "Doctor Account", "appointment_id": 22}	f	2025-11-20 08:00:10.618+07
29	30	APPOINTMENT_REMINDER	{"date": "2025-11-20T00:00:00.000Z", "time": "1970-01-01T09:30:00.000Z", "message": "Nhắc nhở: Bạn có lịch hẹn vào ngày mai", "doctor_name": "Doctor Account", "appointment_id": 24}	f	2025-11-20 08:00:10.618+07
30	32	prescription_ready	{"message": "Prescription #27 is ready for dispensing (Invoice #28).", "invoice_id": 28, "patient_id": 27, "patient_name": "áđâsd", "patient_phone": null, "patient_user_id": null, "prescription_id": 27}	f	2025-11-21 15:44:02.757+07
31	2	appointment_checkin	{"timeslot_id": 993, "patient_name": "Nguyễn Thanh Tú", "appointment_id": 28, "appointment_date": "2025-11-26T00:00:00.000Z", "appointment_time": "1970-01-01T08:00:00.000Z"}	f	2025-11-21 21:37:03.88+07
32	5	appointment_checkin	{"timeslot_id": 1, "patient_name": "Patient One", "appointment_id": 30, "appointment_date": "2025-11-26T00:00:00.000Z", "appointment_time": "1970-01-01T08:00:00.000Z"}	f	2025-11-21 23:24:50.781+07
33	32	prescription_ready	{"message": "Prescription #28 is ready for dispensing (Invoice #29).", "invoice_id": 29, "patient_id": 15, "patient_name": "Nguyễn Thanh Tú", "patient_phone": null, "patient_user_id": null, "prescription_id": 28}	f	2025-11-21 23:28:49.612+07
\.


--
-- TOC entry 5217 (class 0 OID 31301)
-- Dependencies: 230
-- Data for Name: patients; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.patients (id, user_id, gender, blood_type, allergies, emergency_contact, created_at, updated_at, address, owner_user_id, full_name, phone, email, dob, occupation, id_type, id_number, nationality, ethnicity, old_province, old_district, old_ward, old_street, new_province, new_district, new_ward, new_street, id_issue_date, id_issue_place, zalo) FROM stdin;
30	31	Male	\N	\N	{}	2025-11-18 17:24:50.573+07	2025-11-18 17:24:50.573+07	áđâsđá	\N	moi	0786542323	honma953@gmail.com	2025-11-05	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
14	\N	MALE	\N	\N	{}	2025-11-18 13:47:10.519534+07	2025-11-18 13:47:10.519534+07	áđâsd	4	abc	0897786543	thanhtu200220204@gmail.com	2025-11-05	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
15	\N	MALE	\N	\N	{}	2025-11-18 14:26:35.472418+07	2025-11-18 14:26:35.472418+07	389 Nguyễn Ái Quốc	4	Nguyễn Thanh Tú	0785245277	thanhtu200220204@gmail.com	2004-02-20	STUDENT	CCCD	075204011724	Vietnam	Kinh	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
19	\N	Male	\N	\N	{}	2025-11-18 16:53:16.661794+07	2025-11-18 16:53:16.661794+07	áđâsđá	\N	bcxzczxc	1231231231	\N	2025-11-12	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
20	\N	Male	\N	\N	{}	2025-11-18 16:53:59.113959+07	2025-11-18 16:53:59.113959+07	áđâsđâsd	\N	bbbbb	0345612341	\N	2025-11-17	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
21	\N	Male	\N	\N	{}	2025-11-18 16:54:19.919144+07	2025-11-18 16:54:19.919144+07	1áđâsd	\N	svâsđâsd	1232131231	\N	2025-11-12	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
22	\N	Male	\N	\N	{}	2025-11-18 16:57:32.697793+07	2025-11-18 16:57:32.697793+07	ádâsđá	\N	bbbb	0123123123	\N	2025-11-10	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
23	\N	Male	\N	\N	{}	2025-11-18 17:03:41.124505+07	2025-11-18 17:03:41.124505+07	đasadsadsad	\N	bbbb	0987654321	\N	2025-11-03	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
24	\N	Male	\N	\N	{}	2025-11-18 17:04:24.33046+07	2025-11-18 17:04:24.33046+07	sađâsd	\N	ádsađá	1231231231	thanhtu20022021312304@gmail.com	2025-11-06	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
25	\N	Male	\N	\N	{}	2025-11-18 17:05:30.247607+07	2025-11-18 17:05:30.247607+07	áđâsd	\N	sad	1233123123	test123123@email.com	2025-11-09	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
27	\N	Male	\N	\N	{}	2025-11-18 17:12:21.689686+07	2025-11-18 17:12:21.689686+07	123123	\N	áđâsd	012356789	\N	2025-11-17	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
28	\N	Male	\N	\N	{}	2025-11-18 17:15:05.96871+07	2025-11-18 17:15:05.96871+07	ádxxvêrêr	\N	mới	0981234567	\N	2025-11-03	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
29	\N	Male	\N	\N	{}	2025-11-18 17:20:31.068847+07	2025-11-18 17:20:31.068847+07	áđâsđá	\N	zxcxzczxc	0984563271	\N	2025-11-04	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
1	9	M	O+	None	\N	2025-11-16 17:09:31.391+07	2025-11-16 17:09:31.391+07	123 Main St, City	9	Patient One	\N	patient1@email.com	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
2	10	F	A+	None	\N	2025-11-16 17:09:31.45+07	2025-11-16 17:09:31.45+07	123 Main St, City	10	Patient Two	\N	patient2@email.com	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
3	11	M	B+	None	\N	2025-11-16 17:09:31.508+07	2025-11-16 17:09:31.508+07	123 Main St, City	11	Patient Three	\N	patient3@email.com	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
4	12	F	AB+	None	\N	2025-11-16 17:09:31.566+07	2025-11-16 17:09:31.566+07	123 Main St, City	12	Patient Four	\N	patient4@email.com	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
6	22	M	O+	None	\N	2025-11-17 19:42:38.087+07	2025-11-17 19:42:38.087+07	123 Main St, City	22	Patient One	\N	patient1@example.test	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
7	23	F	A+	None	\N	2025-11-17 19:42:38.47+07	2025-11-17 19:42:38.47+07	123 Main St, City	23	Patient Two	\N	patient2@example.test	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
8	24	M	B+	None	\N	2025-11-17 19:42:38.698+07	2025-11-17 19:42:38.698+07	123 Main St, City	24	Patient Three	\N	patient3@example.test	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
9	25	F	AB+	None	\N	2025-11-17 19:42:39.015+07	2025-11-17 19:42:39.015+07	123 Main St, City	25	Patient Four	\N	patient4@example.test	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
10	26	Male	\N	\N	{}	2025-11-17 20:34:03.5+07	2025-11-17 20:34:03.5+07	123 Main St, City	26	aaa	0123123123	patient+1763386443076@example.test	2025-11-12	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
5	4	MALE	O+	None	\N	2025-11-16 18:55:12.881+07	2025-11-16 18:55:12.881+07	aaaaaâ	4	Patient Account111	0785235277	patient@email.com	2025-11-05	STUDENT	CCCD	1231231231	Vietnam	kinh	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
16	27	\N	\N	\N	{}	2025-11-18 16:04:41.727+07	2025-11-18 16:04:41.727+07	123 Main St, City	\N	tu	0123456789	tes111t@email.com	2025-11-11	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
17	28	Male	\N	\N	{}	2025-11-18 16:37:57.369+07	2025-11-18 16:37:57.369+07	123 Main St, City	\N	áđá	0213123123	test1121311@email.com	2025-11-05	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
18	29	Male	\N	\N	{}	2025-11-18 16:41:22.963+07	2025-11-18 16:41:22.963+07	123 Main St, City	\N	ádsad	0123123123	patient+1763458882835@example.test	2025-11-12	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
26	30	\N	\N	\N	{}	2025-11-18 17:07:20.877+07	2025-11-18 17:07:20.877+07	123 Main St, City	\N	àqưeãzc	0987654321	tes1121t@email.com	2025-11-11	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	\N	f
\.


--
-- TOC entry 5233 (class 0 OID 31389)
-- Dependencies: 246
-- Data for Name: prescriptions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.prescriptions (id, appointment_id, doctor_id, patient_id, items, total_amount, status, created_at, updated_at, notified_at, notified_by) FROM stdin;
1	1	1	1	[{"quantity": 10, "unit_price": 10000, "medicine_id": 1, "medicine_name": "Paracetamol"}, {"quantity": 5, "unit_price": 15000, "medicine_id": 2, "medicine_name": "Ibuprofen"}]	175000	APPROVED	2025-11-16 17:09:31.793+07	2025-11-16 17:09:31.793+07	\N	\N
2	2	1	2	[{"quantity": 10, "unit_price": 10000, "medicine_id": 1, "medicine_name": "Paracetamol"}, {"quantity": 5, "unit_price": 15000, "medicine_id": 2, "medicine_name": "Ibuprofen"}]	175000	APPROVED	2025-11-16 17:09:31.802+07	2025-11-16 17:09:31.802+07	\N	\N
3	3	1	3	[{"quantity": 10, "unit_price": 10000, "medicine_id": 1, "medicine_name": "Paracetamol"}, {"quantity": 5, "unit_price": 15000, "medicine_id": 2, "medicine_name": "Ibuprofen"}]	175000	APPROVED	2025-11-16 17:09:31.805+07	2025-11-16 17:09:31.805+07	\N	\N
4	4	1	4	[{"quantity": 10, "unit_price": 10000, "medicine_id": 1, "medicine_name": "Paracetamol"}, {"quantity": 5, "unit_price": 15000, "medicine_id": 2, "medicine_name": "Ibuprofen"}]	175000	APPROVED	2025-11-16 17:09:31.807+07	2025-11-16 17:09:31.807+07	\N	\N
5	5	5	5	[{"quantity": 10, "unit_price": 10000, "medicine_id": 1, "medicine_name": "Paracetamol"}, {"quantity": 5, "unit_price": 15000, "medicine_id": 2, "medicine_name": "Ibuprofen"}]	175000	APPROVED	2025-11-17 19:42:42.917+07	2025-11-17 19:42:42.917+07	\N	\N
6	6	5	5	[{"quantity": 10, "unit_price": 10000, "medicine_id": 1, "medicine_name": "Paracetamol"}, {"quantity": 5, "unit_price": 15000, "medicine_id": 2, "medicine_name": "Ibuprofen"}]	175000	APPROVED	2025-11-17 19:42:42.94+07	2025-11-17 19:42:42.94+07	\N	\N
7	7	1	6	[{"quantity": 10, "unit_price": 10000, "medicine_id": 1, "medicine_name": "Paracetamol"}, {"quantity": 5, "unit_price": 15000, "medicine_id": 2, "medicine_name": "Ibuprofen"}]	175000	APPROVED	2025-11-17 19:42:42.951+07	2025-11-17 19:42:42.951+07	\N	\N
8	8	1	7	[{"quantity": 10, "unit_price": 10000, "medicine_id": 1, "medicine_name": "Paracetamol"}, {"quantity": 5, "unit_price": 15000, "medicine_id": 2, "medicine_name": "Ibuprofen"}]	175000	APPROVED	2025-11-17 19:42:42.964+07	2025-11-17 19:42:42.964+07	\N	\N
9	9	1	8	[{"quantity": 10, "unit_price": 10000, "medicine_id": 1, "medicine_name": "Paracetamol"}, {"quantity": 5, "unit_price": 15000, "medicine_id": 2, "medicine_name": "Ibuprofen"}]	175000	APPROVED	2025-11-17 19:42:42.975+07	2025-11-17 19:42:42.975+07	\N	\N
10	10	1	9	[{"quantity": 10, "unit_price": 10000, "medicine_id": 1, "medicine_name": "Paracetamol"}, {"quantity": 5, "unit_price": 15000, "medicine_id": 2, "medicine_name": "Ibuprofen"}]	175000	APPROVED	2025-11-17 19:42:42.991+07	2025-11-17 19:42:42.991+07	\N	\N
11	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 10000, "medicine_id": 1, "instructions": "áđâsđâsd", "medicine_name": "Paracetamol"}]	20000	DRAFT	2025-11-19 16:30:26.075+07	2025-11-19 16:30:26.075+07	\N	\N
12	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 12000, "medicine_id": 4, "instructions": "áđâsđâsd", "medicine_name": "Aspirin"}]	24000	DRAFT	2025-11-19 17:14:24.553+07	2025-11-19 17:14:24.553+07	\N	\N
13	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 12000, "medicine_id": 4, "instructions": "áđâsđâsd", "medicine_name": "Aspirin"}]	24000	DRAFT	2025-11-19 17:16:07.224+07	2025-11-19 17:16:09.368+07	\N	\N
14	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 12000, "medicine_id": 4, "instructions": "áđâsd", "medicine_name": "Aspirin"}]	24000	DRAFT	2025-11-19 17:16:57.605+07	2025-11-19 17:16:59.556+07	\N	\N
15	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 10000, "medicine_id": 1, "instructions": "áđâsđâsd", "medicine_name": "Paracetamol"}]	20000	DRAFT	2025-11-19 17:28:38.369+07	2025-11-19 17:28:41.038+07	\N	\N
16	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 25000, "medicine_id": 3, "instructions": "aaaaaaaaaaa", "medicine_name": "Amoxicillin"}]	50000	DRAFT	2025-11-19 17:43:15.864+07	2025-11-19 17:43:17.363+07	\N	\N
17	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 12000, "medicine_id": 4, "instructions": "áđâsđâsd", "medicine_name": "Aspirin"}]	24000	DRAFT	2025-11-19 17:59:40.868+07	2025-11-19 17:59:42.29+07	\N	\N
18	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 12000, "medicine_id": 4, "instructions": "áddddddđ", "medicine_name": "Aspirin"}]	24000	DRAFT	2025-11-19 18:02:30.553+07	2025-11-19 18:02:39.151+07	\N	\N
19	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 12000, "medicine_id": 4, "instructions": "sađâsđâsd", "medicine_name": "Aspirin"}]	24000	DRAFT	2025-11-20 00:41:40.203+07	2025-11-20 00:41:42.019+07	2025-11-20 00:41:42.019+07	2
20	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 12000, "medicine_id": 4, "instructions": "áđâsđasad", "medicine_name": "Aspirin"}]	24000	DRAFT	2025-11-20 04:47:07.182+07	2025-11-20 04:47:09.053+07	2025-11-20 04:47:09.053+07	2
21	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 12000, "medicine_id": 4, "instructions": "áddddddddddd", "medicine_name": "Aspirin"}]	24000	DRAFT	2025-11-20 04:48:24.76+07	2025-11-20 04:48:26.542+07	2025-11-20 04:48:26.542+07	2
22	\N	5	26	[{"dosage": "1", "quantity": 2, "unit_price": 25000, "medicine_id": 3, "instructions": "áđâsđá", "medicine_name": "Amoxicillin"}]	50000	DRAFT	2025-11-20 04:57:48.137+07	2025-11-20 04:57:50.116+07	2025-11-20 04:57:50.116+07	2
23	\N	5	27	[{"dosage": "1", "quantity": 2, "unit_price": 25000, "medicine_id": 3, "instructions": "áddddddddd", "medicine_name": "Amoxicillin"}]	50000	DRAFT	2025-11-20 05:01:20.354+07	2025-11-20 05:01:21.987+07	2025-11-20 05:01:21.987+07	2
24	\N	5	26	[{"dosage": "1", "quantity": 6, "unit_price": 25000, "medicine_id": 3, "instructions": "áddddddddd", "medicine_name": "Amoxicillin"}]	150000	DISPENSED	2025-11-20 05:02:52.252+07	2025-11-20 05:43:31.665+07	2025-11-20 05:02:54.136+07	2
25	\N	5	26	[{"dosage": "5", "quantity": 20, "unit_price": 10000, "medicine_id": 1, "instructions": "abqưezczxc", "medicine_name": "Paracetamol"}]	200000	DISPENSED	2025-11-20 05:54:37.728+07	2025-11-20 05:55:22.992+07	2025-11-20 05:54:39.27+07	2
26	\N	5	1	[{"dosage": "5", "quantity": 50, "unit_price": 12000, "medicine_id": 4, "instructions": "áđâsđá", "medicine_name": "Aspirin"}, {"dosage": "3", "quantity": 30, "unit_price": 10000, "medicine_id": 1, "instructions": "vxvcáđá", "medicine_name": "Paracetamol"}]	900000	DISPENSED	2025-11-20 06:05:52.708+07	2025-11-20 06:06:39.692+07	2025-11-20 06:05:54.551+07	2
27	\N	5	27	[{"dosage": "1", "quantity": 2, "unit_price": 10000, "medicine_id": 1, "instructions": "ko uong", "medicine_name": "Paracetamol"}]	20000	DISPENSED	2025-11-21 15:43:14.893+07	2025-11-21 15:47:39.601+07	2025-11-21 15:43:23.289+07	2
28	\N	5	15	[{"dosage": "1", "quantity": 2, "unit_price": 10000, "medicine_id": 1, "instructions": "aaaa", "medicine_name": "Paracetamol"}]	20000	DISPENSED	2025-11-21 23:27:58.76+07	2025-11-21 23:29:57.878+07	2025-11-21 23:28:00.488+07	2
\.


--
-- TOC entry 5207 (class 0 OID 31248)
-- Dependencies: 220
-- Data for Name: roles; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.roles (id, name, description, created_at) FROM stdin;
1	Admin	Administrator	2025-11-16 17:09:30.848+07
2	Doctor	Doctor role	2025-11-16 17:09:30.853+07
3	Reception	Reception staff	2025-11-16 17:09:30.854+07
4	Patient	Patient role	2025-11-16 17:09:30.855+07
5	Receptionist	Reception staff	2025-11-17 19:58:23.46+07
6	Pharmacist	Dược sĩ	2025-11-20 03:57:33.536886+07
\.


--
-- TOC entry 5219 (class 0 OID 31312)
-- Dependencies: 232
-- Data for Name: rooms; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.rooms (id, name, type, description, capacity, created_at, doctor_id) FROM stdin;
1	Phòng bác sĩ 1 - Dr. Nguyen Van A	Doctor	\N	1	2025-11-16 17:09:31.578+07	1
2	Phòng bác sĩ 2 - Dr. Tran Thi B	Doctor	\N	1	2025-11-16 17:09:31.582+07	2
3	Phòng bác sĩ 3 - Dr. Le Van C	Doctor	\N	1	2025-11-16 17:09:31.584+07	3
4	Phòng bác sĩ 4 - Dr. Pham Thi D	Doctor	\N	1	2025-11-16 17:09:31.586+07	4
5	Phòng Lab 1	Lab	\N	5	2025-11-16 17:09:31.586+07	\N
6	Phòng Lab 2	Lab	\N	5	2025-11-16 17:09:31.586+07	\N
7	Quầy lễ tân 1	Reception	\N	10	2025-11-16 17:09:31.586+07	\N
8	Quầy lễ tân 2	Reception	\N	10	2025-11-16 17:09:31.586+07	\N
9	Phòng bác sĩ 6 - Dr. Nguyen Van A	Doctor	\N	1	2025-11-17 19:42:39.247+07	6
10	Phòng bác sĩ 7 - Dr. Tran Thi B	Doctor	\N	1	2025-11-17 19:42:39.263+07	7
11	Phòng bác sĩ 8 - Dr. Le Van C	Doctor	\N	1	2025-11-17 19:42:39.287+07	8
12	Phòng bác sĩ 9 - Dr. Pham Thi D	Doctor	\N	1	2025-11-17 19:42:39.318+07	9
\.


--
-- TOC entry 5221 (class 0 OID 31323)
-- Dependencies: 234
-- Data for Name: schedules; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.schedules (id, doctor_id, room_id, date, start_time, end_time, recurrent_rule, created_at) FROM stdin;
7	1	1	2025-11-21	08:00:00	17:00:00	\N	2025-11-16 17:09:31.637+07
14	2	2	2025-11-21	08:00:00	17:00:00	\N	2025-11-16 17:09:31.675+07
21	3	3	2025-11-21	08:00:00	17:00:00	\N	2025-11-16 17:09:31.728+07
28	4	4	2025-11-21	08:00:00	17:00:00	\N	2025-11-16 17:09:31.767+07
37	6	9	2025-11-21	01:00:00	10:00:00	\N	2025-11-17 19:42:40.014+07
38	6	9	2025-11-22	01:00:00	10:00:00	\N	2025-11-17 19:42:40.122+07
44	7	10	2025-11-21	01:00:00	10:00:00	\N	2025-11-17 19:42:40.877+07
45	7	10	2025-11-22	01:00:00	10:00:00	\N	2025-11-17 19:42:40.981+07
51	8	11	2025-11-21	01:00:00	10:00:00	\N	2025-11-17 19:42:41.631+07
52	8	11	2025-11-22	01:00:00	10:00:00	\N	2025-11-17 19:42:41.725+07
58	9	12	2025-11-21	01:00:00	10:00:00	\N	2025-11-17 19:42:42.378+07
59	9	12	2025-11-22	01:00:00	10:00:00	\N	2025-11-17 19:42:42.478+07
61	6	9	2025-11-23	01:00:00	10:00:00	\N	2025-11-17 22:09:24.677+07
62	7	10	2025-11-23	01:00:00	10:00:00	\N	2025-11-17 22:09:40.81+07
63	8	11	2025-11-23	01:00:00	10:00:00	\N	2025-11-17 22:09:40.874+07
64	9	12	2025-11-23	01:00:00	10:00:00	\N	2025-11-17 22:09:40.937+07
67	5	\N	2025-11-22	08:00:00	12:00:00	\N	2025-11-21 15:31:46.935+07
68	1	\N	2025-11-23	08:00:00	17:00:00	\N	2025-11-21 16:37:09.244+07
69	5	\N	2025-11-22	08:00:00	17:00:00	\N	2025-11-21 16:49:17.33+07
116	5	\N	2025-11-26	08:00:00	11:00:00	\N	2025-11-21 18:51:49.807+07
117	6	9	2025-11-24	01:00:00	10:00:00	\N	2025-11-21 20:10:51.834+07
118	6	9	2025-11-25	01:00:00	10:00:00	\N	2025-11-21 20:10:51.899+07
119	6	9	2025-11-26	01:00:00	10:00:00	\N	2025-11-21 20:10:51.969+07
120	7	10	2025-11-24	01:00:00	10:00:00	\N	2025-11-21 20:10:52.11+07
121	7	10	2025-11-25	01:00:00	10:00:00	\N	2025-11-21 20:10:52.153+07
122	7	10	2025-11-26	01:00:00	10:00:00	\N	2025-11-21 20:10:52.193+07
123	8	11	2025-11-24	01:00:00	10:00:00	\N	2025-11-21 20:10:52.316+07
124	8	11	2025-11-25	01:00:00	10:00:00	\N	2025-11-21 20:10:52.362+07
125	8	11	2025-11-26	01:00:00	10:00:00	\N	2025-11-21 20:10:52.408+07
126	9	12	2025-11-24	01:00:00	10:00:00	\N	2025-11-21 20:10:52.529+07
127	9	12	2025-11-25	01:00:00	10:00:00	\N	2025-11-21 20:10:52.569+07
128	9	12	2025-11-26	01:00:00	10:00:00	\N	2025-11-21 20:10:52.61+07
\.


--
-- TOC entry 5213 (class 0 OID 31282)
-- Dependencies: 226
-- Data for Name: specialties; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.specialties (id, name, slug, description, image_url, created_at, updated_at) FROM stdin;
1	General	general	\N	\N	2025-11-16 17:09:31.567+07	2025-11-16 17:09:31.567+07
2	Cardiology	cardiology	\N	\N	2025-11-16 17:09:31.569+07	2025-11-16 17:09:31.569+07
3	Dermatology	dermatology	\N	\N	2025-11-16 17:09:31.569+07	2025-11-16 17:09:31.569+07
4	Pediatrics	pediatrics	\N	\N	2025-11-16 17:09:31.57+07	2025-11-16 17:09:31.57+07
5	Orthopedics	orthopedics	\N	\N	2025-11-16 17:09:31.57+07	2025-11-16 17:09:31.57+07
\.


--
-- TOC entry 5231 (class 0 OID 31379)
-- Dependencies: 244
-- Data for Name: stocks; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.stocks (id, medicine_id, batch_number, expiry_date, quantity, created_at, updated_at) FROM stdin;
2	2	BATCH-001	2026-12-30	100	2025-11-16 17:09:31.784+07	2025-11-16 17:09:31.784+07
3	3	BATCH-001	2026-12-30	94	2025-11-16 17:09:31.785+07	2025-11-20 05:43:31.653+07
4	4	BATCH-001	2026-12-30	50	2025-11-16 17:09:31.787+07	2025-11-20 06:06:39.685+07
1	1	BATCH-001	2026-12-30	46	2025-11-16 17:09:31.782+07	2025-11-21 23:29:57.874+07
\.


--
-- TOC entry 5235 (class 0 OID 31402)
-- Dependencies: 248
-- Data for Name: suppliers; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.suppliers (id, name, contact_info, created_at) FROM stdin;
1	ABC Pharma	{"phone": "0123456789"}	2025-11-16 17:09:31.788+07
2	XYZ Medical Supply	{"phone": "0987654321"}	2025-11-16 17:09:31.788+07
3	Global Healthcare	{"phone": "0111222333"}	2025-11-16 17:09:31.788+07
\.


--
-- TOC entry 5223 (class 0 OID 31331)
-- Dependencies: 236
-- Data for Name: timeslots; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.timeslots (id, doctor_id, date, start_time, end_time, max_patients, booked_count, is_active, created_at, updated_at) FROM stdin;
402	7	2025-11-21	01:00:00	02:00:00	1	0	t	2025-11-17 19:42:40.888+07	2025-11-17 19:42:40.888+07
403	7	2025-11-21	02:00:00	03:00:00	1	0	t	2025-11-17 19:42:40.896+07	2025-11-17 19:42:40.896+07
404	7	2025-11-21	03:00:00	04:00:00	1	0	t	2025-11-17 19:42:40.907+07	2025-11-17 19:42:40.907+07
405	7	2025-11-21	04:00:00	05:00:00	1	0	t	2025-11-17 19:42:40.92+07	2025-11-17 19:42:40.92+07
406	7	2025-11-21	06:00:00	07:00:00	1	0	t	2025-11-17 19:42:40.928+07	2025-11-17 19:42:40.928+07
407	7	2025-11-21	07:00:00	08:00:00	1	0	t	2025-11-17 19:42:40.942+07	2025-11-17 19:42:40.942+07
408	7	2025-11-21	08:00:00	09:00:00	1	0	t	2025-11-17 19:42:40.953+07	2025-11-17 19:42:40.953+07
409	7	2025-11-21	09:00:00	10:00:00	1	0	t	2025-11-17 19:42:40.967+07	2025-11-17 19:42:40.967+07
410	7	2025-11-22	01:00:00	02:00:00	1	0	t	2025-11-17 19:42:40.994+07	2025-11-17 19:42:40.994+07
411	7	2025-11-22	02:00:00	03:00:00	1	0	t	2025-11-17 19:42:41.007+07	2025-11-17 19:42:41.007+07
412	7	2025-11-22	03:00:00	04:00:00	1	0	t	2025-11-17 19:42:41.018+07	2025-11-17 19:42:41.018+07
413	7	2025-11-22	04:00:00	05:00:00	1	0	t	2025-11-17 19:42:41.028+07	2025-11-17 19:42:41.028+07
414	7	2025-11-22	06:00:00	07:00:00	1	0	t	2025-11-17 19:42:41.039+07	2025-11-17 19:42:41.039+07
415	7	2025-11-22	07:00:00	08:00:00	1	0	t	2025-11-17 19:42:41.047+07	2025-11-17 19:42:41.047+07
416	7	2025-11-22	08:00:00	09:00:00	1	0	t	2025-11-17 19:42:41.056+07	2025-11-17 19:42:41.056+07
417	7	2025-11-22	09:00:00	10:00:00	1	0	t	2025-11-17 19:42:41.068+07	2025-11-17 19:42:41.068+07
458	8	2025-11-21	01:00:00	02:00:00	1	0	t	2025-11-17 19:42:41.641+07	2025-11-17 19:42:41.641+07
459	8	2025-11-21	02:00:00	03:00:00	1	0	t	2025-11-17 19:42:41.653+07	2025-11-17 19:42:41.653+07
460	8	2025-11-21	03:00:00	04:00:00	1	0	t	2025-11-17 19:42:41.663+07	2025-11-17 19:42:41.663+07
461	8	2025-11-21	04:00:00	05:00:00	1	0	t	2025-11-17 19:42:41.673+07	2025-11-17 19:42:41.673+07
462	8	2025-11-21	06:00:00	07:00:00	1	0	t	2025-11-17 19:42:41.681+07	2025-11-17 19:42:41.681+07
463	8	2025-11-21	07:00:00	08:00:00	1	0	t	2025-11-17 19:42:41.691+07	2025-11-17 19:42:41.691+07
464	8	2025-11-21	08:00:00	09:00:00	1	0	t	2025-11-17 19:42:41.702+07	2025-11-17 19:42:41.702+07
465	8	2025-11-21	09:00:00	10:00:00	1	0	t	2025-11-17 19:42:41.715+07	2025-11-17 19:42:41.715+07
466	8	2025-11-22	01:00:00	02:00:00	1	0	t	2025-11-17 19:42:41.736+07	2025-11-17 19:42:41.736+07
467	8	2025-11-22	02:00:00	03:00:00	1	0	t	2025-11-17 19:42:41.745+07	2025-11-17 19:42:41.745+07
468	8	2025-11-22	03:00:00	04:00:00	1	0	t	2025-11-17 19:42:41.756+07	2025-11-17 19:42:41.756+07
469	8	2025-11-22	04:00:00	05:00:00	1	0	t	2025-11-17 19:42:41.765+07	2025-11-17 19:42:41.765+07
470	8	2025-11-22	06:00:00	07:00:00	1	0	t	2025-11-17 19:42:41.778+07	2025-11-17 19:42:41.778+07
471	8	2025-11-22	07:00:00	08:00:00	1	0	t	2025-11-17 19:42:41.793+07	2025-11-17 19:42:41.793+07
472	8	2025-11-22	08:00:00	09:00:00	1	0	t	2025-11-17 19:42:41.804+07	2025-11-17 19:42:41.804+07
473	8	2025-11-22	09:00:00	10:00:00	1	0	t	2025-11-17 19:42:41.813+07	2025-11-17 19:42:41.813+07
514	9	2025-11-21	01:00:00	02:00:00	1	0	t	2025-11-17 19:42:42.39+07	2025-11-17 19:42:42.39+07
515	9	2025-11-21	02:00:00	03:00:00	1	0	t	2025-11-17 19:42:42.4+07	2025-11-17 19:42:42.4+07
516	9	2025-11-21	03:00:00	04:00:00	1	0	t	2025-11-17 19:42:42.41+07	2025-11-17 19:42:42.41+07
517	9	2025-11-21	04:00:00	05:00:00	1	0	t	2025-11-17 19:42:42.424+07	2025-11-17 19:42:42.424+07
518	9	2025-11-21	06:00:00	07:00:00	1	0	t	2025-11-17 19:42:42.432+07	2025-11-17 19:42:42.432+07
519	9	2025-11-21	07:00:00	08:00:00	1	0	t	2025-11-17 19:42:42.444+07	2025-11-17 19:42:42.444+07
520	9	2025-11-21	08:00:00	09:00:00	1	0	t	2025-11-17 19:42:42.458+07	2025-11-17 19:42:42.458+07
521	9	2025-11-21	09:00:00	10:00:00	1	0	t	2025-11-17 19:42:42.468+07	2025-11-17 19:42:42.468+07
522	9	2025-11-22	01:00:00	02:00:00	1	0	t	2025-11-17 19:42:42.487+07	2025-11-17 19:42:42.487+07
523	9	2025-11-22	02:00:00	03:00:00	1	0	t	2025-11-17 19:42:42.496+07	2025-11-17 19:42:42.496+07
524	9	2025-11-22	03:00:00	04:00:00	1	0	t	2025-11-17 19:42:42.506+07	2025-11-17 19:42:42.506+07
525	9	2025-11-22	04:00:00	05:00:00	1	0	t	2025-11-17 19:42:42.518+07	2025-11-17 19:42:42.518+07
526	9	2025-11-22	06:00:00	07:00:00	1	0	t	2025-11-17 19:42:42.529+07	2025-11-17 19:42:42.529+07
527	9	2025-11-22	07:00:00	08:00:00	1	0	t	2025-11-17 19:42:42.539+07	2025-11-17 19:42:42.539+07
528	9	2025-11-22	08:00:00	09:00:00	1	0	t	2025-11-17 19:42:42.547+07	2025-11-17 19:42:42.547+07
529	9	2025-11-22	09:00:00	10:00:00	1	0	t	2025-11-17 19:42:42.556+07	2025-11-17 19:42:42.556+07
542	6	2025-11-23	01:00:00	02:00:00	1	0	t	2025-11-17 22:09:40.736+07	2025-11-17 22:09:40.736+07
543	6	2025-11-23	02:00:00	03:00:00	1	0	t	2025-11-17 22:09:40.751+07	2025-11-17 22:09:40.751+07
544	6	2025-11-23	03:00:00	04:00:00	1	0	t	2025-11-17 22:09:40.753+07	2025-11-17 22:09:40.753+07
545	6	2025-11-23	04:00:00	05:00:00	1	0	t	2025-11-17 22:09:40.755+07	2025-11-17 22:09:40.755+07
546	6	2025-11-23	06:00:00	07:00:00	1	0	t	2025-11-17 22:09:40.757+07	2025-11-17 22:09:40.757+07
547	6	2025-11-23	07:00:00	08:00:00	1	0	t	2025-11-17 22:09:40.759+07	2025-11-17 22:09:40.759+07
548	6	2025-11-23	08:00:00	09:00:00	1	0	t	2025-11-17 22:09:40.761+07	2025-11-17 22:09:40.761+07
549	6	2025-11-23	09:00:00	10:00:00	1	0	t	2025-11-17 22:09:40.762+07	2025-11-17 22:09:40.762+07
550	7	2025-11-23	01:00:00	02:00:00	1	0	t	2025-11-17 22:09:40.812+07	2025-11-17 22:09:40.812+07
551	7	2025-11-23	02:00:00	03:00:00	1	0	t	2025-11-17 22:09:40.814+07	2025-11-17 22:09:40.814+07
552	7	2025-11-23	03:00:00	04:00:00	1	0	t	2025-11-17 22:09:40.816+07	2025-11-17 22:09:40.816+07
553	7	2025-11-23	04:00:00	05:00:00	1	0	t	2025-11-17 22:09:40.818+07	2025-11-17 22:09:40.818+07
41	1	2025-11-21	08:00:00	09:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
42	1	2025-11-21	09:00:00	10:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
43	1	2025-11-21	10:00:00	11:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
44	1	2025-11-21	11:00:00	12:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
45	1	2025-11-21	13:00:00	14:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
46	1	2025-11-21	14:00:00	15:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
47	1	2025-11-21	15:00:00	16:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
48	1	2025-11-21	16:00:00	17:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
51	1	2025-11-22	10:00:00	11:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
53	1	2025-11-22	13:00:00	14:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
54	1	2025-11-22	14:00:00	15:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
55	1	2025-11-22	15:00:00	16:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
56	1	2025-11-22	16:00:00	17:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
97	2	2025-11-21	08:00:00	09:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
98	2	2025-11-21	09:00:00	10:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
99	2	2025-11-21	10:00:00	11:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
100	2	2025-11-21	11:00:00	12:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
101	2	2025-11-21	13:00:00	14:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
102	2	2025-11-21	14:00:00	15:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
103	2	2025-11-21	15:00:00	16:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
104	2	2025-11-21	16:00:00	17:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
105	2	2025-11-22	08:00:00	09:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
106	2	2025-11-22	09:00:00	10:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
107	2	2025-11-22	10:00:00	11:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
108	2	2025-11-22	11:00:00	12:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
109	2	2025-11-22	13:00:00	14:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
110	2	2025-11-22	14:00:00	15:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
111	2	2025-11-22	15:00:00	16:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
112	2	2025-11-22	16:00:00	17:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
52	1	2025-11-22	11:00:00	12:00:00	1	1	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
50	1	2025-11-22	09:00:00	10:00:00	1	1	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
153	3	2025-11-21	08:00:00	09:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
154	3	2025-11-21	09:00:00	10:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
155	3	2025-11-21	10:00:00	11:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
156	3	2025-11-21	11:00:00	12:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
157	3	2025-11-21	13:00:00	14:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
158	3	2025-11-21	14:00:00	15:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
159	3	2025-11-21	15:00:00	16:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
160	3	2025-11-21	16:00:00	17:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
161	3	2025-11-22	08:00:00	09:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
162	3	2025-11-22	09:00:00	10:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
163	3	2025-11-22	10:00:00	11:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
164	3	2025-11-22	11:00:00	12:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
165	3	2025-11-22	13:00:00	14:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
166	3	2025-11-22	14:00:00	15:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
167	3	2025-11-22	15:00:00	16:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
168	3	2025-11-22	16:00:00	17:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
209	4	2025-11-21	08:00:00	09:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
210	4	2025-11-21	09:00:00	10:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
211	4	2025-11-21	10:00:00	11:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
212	4	2025-11-21	11:00:00	12:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
213	4	2025-11-21	13:00:00	14:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
214	4	2025-11-21	14:00:00	15:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
215	4	2025-11-21	15:00:00	16:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
216	4	2025-11-21	16:00:00	17:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
217	4	2025-11-22	08:00:00	09:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
218	4	2025-11-22	09:00:00	10:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
219	4	2025-11-22	10:00:00	11:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
220	4	2025-11-22	11:00:00	12:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
221	4	2025-11-22	13:00:00	14:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
222	4	2025-11-22	14:00:00	15:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
223	4	2025-11-22	15:00:00	16:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
224	4	2025-11-22	16:00:00	17:00:00	1	0	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
346	6	2025-11-21	01:00:00	02:00:00	1	0	t	2025-11-17 19:42:40.025+07	2025-11-17 19:42:40.025+07
347	6	2025-11-21	02:00:00	03:00:00	1	0	t	2025-11-17 19:42:40.033+07	2025-11-17 19:42:40.033+07
348	6	2025-11-21	03:00:00	04:00:00	1	0	t	2025-11-17 19:42:40.044+07	2025-11-17 19:42:40.044+07
349	6	2025-11-21	04:00:00	05:00:00	1	0	t	2025-11-17 19:42:40.053+07	2025-11-17 19:42:40.053+07
350	6	2025-11-21	06:00:00	07:00:00	1	0	t	2025-11-17 19:42:40.066+07	2025-11-17 19:42:40.066+07
351	6	2025-11-21	07:00:00	08:00:00	1	0	t	2025-11-17 19:42:40.078+07	2025-11-17 19:42:40.078+07
352	6	2025-11-21	08:00:00	09:00:00	1	0	t	2025-11-17 19:42:40.09+07	2025-11-17 19:42:40.09+07
353	6	2025-11-21	09:00:00	10:00:00	1	0	t	2025-11-17 19:42:40.105+07	2025-11-17 19:42:40.105+07
354	6	2025-11-22	01:00:00	02:00:00	1	0	t	2025-11-17 19:42:40.138+07	2025-11-17 19:42:40.138+07
355	6	2025-11-22	02:00:00	03:00:00	1	0	t	2025-11-17 19:42:40.153+07	2025-11-17 19:42:40.153+07
356	6	2025-11-22	03:00:00	04:00:00	1	0	t	2025-11-17 19:42:40.165+07	2025-11-17 19:42:40.165+07
357	6	2025-11-22	04:00:00	05:00:00	1	0	t	2025-11-17 19:42:40.189+07	2025-11-17 19:42:40.189+07
358	6	2025-11-22	06:00:00	07:00:00	1	0	t	2025-11-17 19:42:40.206+07	2025-11-17 19:42:40.206+07
359	6	2025-11-22	07:00:00	08:00:00	1	0	t	2025-11-17 19:42:40.22+07	2025-11-17 19:42:40.22+07
360	6	2025-11-22	08:00:00	09:00:00	1	0	t	2025-11-17 19:42:40.232+07	2025-11-17 19:42:40.232+07
361	6	2025-11-22	09:00:00	10:00:00	1	0	t	2025-11-17 19:42:40.255+07	2025-11-17 19:42:40.255+07
554	7	2025-11-23	06:00:00	07:00:00	1	0	t	2025-11-17 22:09:40.819+07	2025-11-17 22:09:40.819+07
555	7	2025-11-23	07:00:00	08:00:00	1	0	t	2025-11-17 22:09:40.823+07	2025-11-17 22:09:40.823+07
556	7	2025-11-23	08:00:00	09:00:00	1	0	t	2025-11-17 22:09:40.824+07	2025-11-17 22:09:40.824+07
557	7	2025-11-23	09:00:00	10:00:00	1	0	t	2025-11-17 22:09:40.827+07	2025-11-17 22:09:40.827+07
558	8	2025-11-23	01:00:00	02:00:00	1	0	t	2025-11-17 22:09:40.876+07	2025-11-17 22:09:40.876+07
559	8	2025-11-23	02:00:00	03:00:00	1	0	t	2025-11-17 22:09:40.878+07	2025-11-17 22:09:40.878+07
560	8	2025-11-23	03:00:00	04:00:00	1	0	t	2025-11-17 22:09:40.88+07	2025-11-17 22:09:40.88+07
561	8	2025-11-23	04:00:00	05:00:00	1	0	t	2025-11-17 22:09:40.882+07	2025-11-17 22:09:40.882+07
562	8	2025-11-23	06:00:00	07:00:00	1	0	t	2025-11-17 22:09:40.883+07	2025-11-17 22:09:40.883+07
563	8	2025-11-23	07:00:00	08:00:00	1	0	t	2025-11-17 22:09:40.885+07	2025-11-17 22:09:40.885+07
564	8	2025-11-23	08:00:00	09:00:00	1	0	t	2025-11-17 22:09:40.887+07	2025-11-17 22:09:40.887+07
565	8	2025-11-23	09:00:00	10:00:00	1	0	t	2025-11-17 22:09:40.889+07	2025-11-17 22:09:40.889+07
566	9	2025-11-23	01:00:00	02:00:00	1	0	t	2025-11-17 22:09:40.94+07	2025-11-17 22:09:40.94+07
567	9	2025-11-23	02:00:00	03:00:00	1	0	t	2025-11-17 22:09:40.942+07	2025-11-17 22:09:40.942+07
568	9	2025-11-23	03:00:00	04:00:00	1	0	t	2025-11-17 22:09:40.944+07	2025-11-17 22:09:40.944+07
569	9	2025-11-23	04:00:00	05:00:00	1	0	t	2025-11-17 22:09:40.945+07	2025-11-17 22:09:40.945+07
570	9	2025-11-23	06:00:00	07:00:00	1	0	t	2025-11-17 22:09:40.947+07	2025-11-17 22:09:40.947+07
571	9	2025-11-23	07:00:00	08:00:00	1	0	t	2025-11-17 22:09:40.949+07	2025-11-17 22:09:40.949+07
572	9	2025-11-23	08:00:00	09:00:00	1	0	t	2025-11-17 22:09:40.95+07	2025-11-17 22:09:40.95+07
573	9	2025-11-23	09:00:00	10:00:00	1	0	t	2025-11-17 22:09:40.956+07	2025-11-17 22:09:40.956+07
651	5	2025-11-22	08:00:00	08:30:00	1	1	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
1002	6	2025-11-24	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:51.85+07	2025-11-21 20:10:51.85+07
1003	6	2025-11-24	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:51.859+07	2025-11-21 20:10:51.859+07
1004	6	2025-11-24	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:51.864+07	2025-11-21 20:10:51.864+07
1005	6	2025-11-24	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:51.869+07	2025-11-21 20:10:51.869+07
1006	6	2025-11-24	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:51.878+07	2025-11-21 20:10:51.878+07
1007	6	2025-11-24	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:51.884+07	2025-11-21 20:10:51.884+07
1008	6	2025-11-24	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:51.889+07	2025-11-21 20:10:51.889+07
1009	6	2025-11-24	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:51.894+07	2025-11-21 20:10:51.894+07
1010	6	2025-11-25	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:51.904+07	2025-11-21 20:10:51.904+07
1011	6	2025-11-25	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:51.909+07	2025-11-21 20:10:51.909+07
1012	6	2025-11-25	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:51.914+07	2025-11-21 20:10:51.914+07
1013	6	2025-11-25	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:51.919+07	2025-11-21 20:10:51.919+07
1014	6	2025-11-25	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:51.924+07	2025-11-21 20:10:51.924+07
1015	6	2025-11-25	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:51.935+07	2025-11-21 20:10:51.935+07
1016	6	2025-11-25	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:51.951+07	2025-11-21 20:10:51.951+07
1017	6	2025-11-25	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:51.959+07	2025-11-21 20:10:51.959+07
1018	6	2025-11-26	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:51.974+07	2025-11-21 20:10:51.974+07
1019	6	2025-11-26	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:51.987+07	2025-11-21 20:10:51.987+07
1020	6	2025-11-26	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:51.994+07	2025-11-21 20:10:51.994+07
1021	6	2025-11-26	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:52.001+07	2025-11-21 20:10:52.001+07
1022	6	2025-11-26	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:52.007+07	2025-11-21 20:10:52.007+07
1023	6	2025-11-26	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:52.015+07	2025-11-21 20:10:52.015+07
1024	6	2025-11-26	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:52.021+07	2025-11-21 20:10:52.021+07
1025	6	2025-11-26	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:52.026+07	2025-11-21 20:10:52.026+07
1026	7	2025-11-24	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:52.116+07	2025-11-21 20:10:52.116+07
1027	7	2025-11-24	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:52.121+07	2025-11-21 20:10:52.121+07
1028	7	2025-11-24	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:52.125+07	2025-11-21 20:10:52.125+07
1029	7	2025-11-24	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:52.131+07	2025-11-21 20:10:52.131+07
1030	7	2025-11-24	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:52.136+07	2025-11-21 20:10:52.136+07
1031	7	2025-11-24	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:52.14+07	2025-11-21 20:10:52.14+07
49	1	2025-11-22	08:00:00	09:00:00	1	1	t	2025-11-16 17:13:00.57113+07	2025-11-16 17:13:00.57113+07
1032	7	2025-11-24	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:52.144+07	2025-11-21 20:10:52.144+07
1033	7	2025-11-24	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:52.149+07	2025-11-21 20:10:52.149+07
1034	7	2025-11-25	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:52.157+07	2025-11-21 20:10:52.157+07
1035	7	2025-11-25	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:52.162+07	2025-11-21 20:10:52.162+07
1036	7	2025-11-25	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:52.167+07	2025-11-21 20:10:52.167+07
1037	7	2025-11-25	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:52.171+07	2025-11-21 20:10:52.171+07
1038	7	2025-11-25	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:52.175+07	2025-11-21 20:10:52.175+07
1039	7	2025-11-25	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:52.18+07	2025-11-21 20:10:52.18+07
1040	7	2025-11-25	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:52.185+07	2025-11-21 20:10:52.185+07
653	5	2025-11-22	09:00:00	09:30:00	1	1	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
654	5	2025-11-22	09:30:00	10:00:00	1	1	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
655	5	2025-11-22	10:00:00	10:30:00	1	1	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
656	5	2025-11-22	10:30:00	11:00:00	1	1	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
657	5	2025-11-22	11:00:00	11:30:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
658	5	2025-11-22	11:30:00	12:00:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
659	5	2025-11-22	12:00:00	12:30:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
660	5	2025-11-22	12:30:00	13:00:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
661	5	2025-11-22	13:00:00	13:30:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
662	5	2025-11-22	13:30:00	14:00:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
663	5	2025-11-22	14:00:00	14:30:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
664	5	2025-11-22	14:30:00	15:00:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
665	5	2025-11-22	15:00:00	15:30:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
666	5	2025-11-22	15:30:00	16:00:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
667	5	2025-11-22	16:00:00	16:30:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
668	5	2025-11-22	16:30:00	17:00:00	1	0	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
994	5	2025-11-26	08:20:00	08:40:00	1	0	t	2025-11-21 18:51:49.82+07	2025-11-21 18:51:49.82+07
995	5	2025-11-26	08:40:00	09:00:00	1	0	t	2025-11-21 18:51:49.823+07	2025-11-21 18:51:49.823+07
996	5	2025-11-26	09:00:00	09:20:00	1	0	t	2025-11-21 18:51:49.826+07	2025-11-21 18:51:49.826+07
997	5	2025-11-26	09:20:00	09:40:00	1	0	t	2025-11-21 18:51:49.83+07	2025-11-21 18:51:49.83+07
998	5	2025-11-26	09:40:00	10:00:00	1	0	t	2025-11-21 18:51:49.832+07	2025-11-21 18:51:49.832+07
999	5	2025-11-26	10:00:00	10:20:00	1	0	t	2025-11-21 18:51:49.835+07	2025-11-21 18:51:49.835+07
1000	5	2025-11-26	10:20:00	10:40:00	1	0	t	2025-11-21 18:51:49.838+07	2025-11-21 18:51:49.838+07
1001	5	2025-11-26	10:40:00	11:00:00	1	0	t	2025-11-21 18:51:49.84+07	2025-11-21 18:51:49.84+07
993	5	2025-11-26	08:00:00	08:20:00	1	1	t	2025-11-21 18:51:49.817+07	2025-11-21 18:51:49.817+07
1041	7	2025-11-25	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:52.189+07	2025-11-21 20:10:52.189+07
1042	7	2025-11-26	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:52.197+07	2025-11-21 20:10:52.197+07
1043	7	2025-11-26	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:52.201+07	2025-11-21 20:10:52.201+07
1044	7	2025-11-26	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:52.206+07	2025-11-21 20:10:52.206+07
1045	7	2025-11-26	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:52.21+07	2025-11-21 20:10:52.21+07
1046	7	2025-11-26	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:52.215+07	2025-11-21 20:10:52.215+07
1047	7	2025-11-26	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:52.219+07	2025-11-21 20:10:52.219+07
1048	7	2025-11-26	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:52.224+07	2025-11-21 20:10:52.224+07
1049	7	2025-11-26	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:52.228+07	2025-11-21 20:10:52.228+07
1050	8	2025-11-24	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:52.321+07	2025-11-21 20:10:52.321+07
1051	8	2025-11-24	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:52.328+07	2025-11-21 20:10:52.328+07
1052	8	2025-11-24	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:52.334+07	2025-11-21 20:10:52.334+07
1053	8	2025-11-24	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:52.339+07	2025-11-21 20:10:52.339+07
1054	8	2025-11-24	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:52.343+07	2025-11-21 20:10:52.343+07
1055	8	2025-11-24	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:52.348+07	2025-11-21 20:10:52.348+07
1056	8	2025-11-24	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:52.353+07	2025-11-21 20:10:52.353+07
1057	8	2025-11-24	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:52.357+07	2025-11-21 20:10:52.357+07
1058	8	2025-11-25	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:52.368+07	2025-11-21 20:10:52.368+07
1059	8	2025-11-25	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:52.373+07	2025-11-21 20:10:52.373+07
1060	8	2025-11-25	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:52.378+07	2025-11-21 20:10:52.378+07
1061	8	2025-11-25	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:52.383+07	2025-11-21 20:10:52.383+07
1062	8	2025-11-25	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:52.387+07	2025-11-21 20:10:52.387+07
1063	8	2025-11-25	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:52.393+07	2025-11-21 20:10:52.393+07
1064	8	2025-11-25	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:52.398+07	2025-11-21 20:10:52.398+07
1065	8	2025-11-25	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:52.403+07	2025-11-21 20:10:52.403+07
1066	8	2025-11-26	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:52.413+07	2025-11-21 20:10:52.413+07
1067	8	2025-11-26	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:52.418+07	2025-11-21 20:10:52.418+07
1068	8	2025-11-26	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:52.423+07	2025-11-21 20:10:52.423+07
1069	8	2025-11-26	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:52.427+07	2025-11-21 20:10:52.427+07
1070	8	2025-11-26	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:52.433+07	2025-11-21 20:10:52.433+07
1071	8	2025-11-26	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:52.438+07	2025-11-21 20:10:52.438+07
1072	8	2025-11-26	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:52.443+07	2025-11-21 20:10:52.443+07
1073	8	2025-11-26	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:52.448+07	2025-11-21 20:10:52.448+07
1074	9	2025-11-24	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:52.534+07	2025-11-21 20:10:52.534+07
1075	9	2025-11-24	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:52.539+07	2025-11-21 20:10:52.539+07
1076	9	2025-11-24	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:52.543+07	2025-11-21 20:10:52.543+07
1077	9	2025-11-24	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:52.548+07	2025-11-21 20:10:52.548+07
1078	9	2025-11-24	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:52.552+07	2025-11-21 20:10:52.552+07
1079	9	2025-11-24	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:52.556+07	2025-11-21 20:10:52.556+07
1080	9	2025-11-24	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:52.56+07	2025-11-21 20:10:52.56+07
1081	9	2025-11-24	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:52.565+07	2025-11-21 20:10:52.565+07
1082	9	2025-11-25	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:52.573+07	2025-11-21 20:10:52.573+07
1083	9	2025-11-25	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:52.578+07	2025-11-21 20:10:52.578+07
1084	9	2025-11-25	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:52.583+07	2025-11-21 20:10:52.583+07
1085	9	2025-11-25	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:52.588+07	2025-11-21 20:10:52.588+07
1086	9	2025-11-25	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:52.592+07	2025-11-21 20:10:52.592+07
1087	9	2025-11-25	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:52.597+07	2025-11-21 20:10:52.597+07
1088	9	2025-11-25	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:52.602+07	2025-11-21 20:10:52.602+07
1089	9	2025-11-25	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:52.606+07	2025-11-21 20:10:52.606+07
1090	9	2025-11-26	01:00:00	02:00:00	1	0	t	2025-11-21 20:10:52.615+07	2025-11-21 20:10:52.615+07
1091	9	2025-11-26	02:00:00	03:00:00	1	0	t	2025-11-21 20:10:52.62+07	2025-11-21 20:10:52.62+07
1092	9	2025-11-26	03:00:00	04:00:00	1	0	t	2025-11-21 20:10:52.625+07	2025-11-21 20:10:52.625+07
1093	9	2025-11-26	04:00:00	05:00:00	1	0	t	2025-11-21 20:10:52.63+07	2025-11-21 20:10:52.63+07
1094	9	2025-11-26	06:00:00	07:00:00	1	0	t	2025-11-21 20:10:52.634+07	2025-11-21 20:10:52.634+07
1095	9	2025-11-26	07:00:00	08:00:00	1	0	t	2025-11-21 20:10:52.639+07	2025-11-21 20:10:52.639+07
1096	9	2025-11-26	08:00:00	09:00:00	1	0	t	2025-11-21 20:10:52.645+07	2025-11-21 20:10:52.645+07
1097	9	2025-11-26	09:00:00	10:00:00	1	0	t	2025-11-21 20:10:52.65+07	2025-11-21 20:10:52.65+07
652	5	2025-11-22	08:30:00	09:00:00	1	1	t	2025-11-21 16:59:03.791+07	2025-11-21 16:59:03.791+07
\.


--
-- TOC entry 5209 (class 0 OID 31258)
-- Dependencies: 222
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (id, email, password, full_name, phone, dob, avatar_url, is_active, role_id, created_at, updated_at) FROM stdin;
1	admin@email.com	$2a$10$nYec6d4Npq0L0kQHin/BBeT/lAZyj1ARhjK91F8qg0UopFPBQmIni	Administrator	\N	\N	\N	t	1	2025-11-16 17:09:30.918+07	2025-11-16 17:09:30.918+07
3	reception@email.com	$2a$10$FdcftgwBnYSx22/uB71Wx.ri4DyWaeWjACeyS50hgoawXd5QDGLS2	Reception Account	\N	\N	\N	t	3	2025-11-16 17:09:31.035+07	2025-11-16 17:09:31.035+07
5	dr1@email.com	$2a$10$JLYtt8Lr1ht5t9.n853xTOaalHqu4a5g4jIPpmMbG54yVa1DcfZn2	Dr. Nguyen Van A	\N	\N	\N	t	2	2025-11-16 17:09:31.152+07	2025-11-16 17:09:31.152+07
6	dr2@email.com	$2a$10$9myac4d/34hJHfD4olb0MOBtKacpyy5if/30Qr2EOJ2Ds.jbEXYSy	Dr. Tran Thi B	\N	\N	\N	t	2	2025-11-16 17:09:31.213+07	2025-11-16 17:09:31.213+07
8	dr4@email.com	$2a$10$nrnrUGKgb4qzZcSqzhGZY.7qZugp.hSuPq2X.Sd4/i3yOlCje/cIK	Dr. Pham Thi D	0785245277	\N	\N	t	2	2025-11-16 17:09:31.331+07	2025-11-16 17:22:28.927+07
9	patient1@email.com	$2a$10$qien94xsugqV0XEdcFNB0OfpbWJctdLDmql6ARQOxeKBuIJvPOHDK	Patient One	\N	\N	\N	t	4	2025-11-16 17:09:31.39+07	2025-11-16 17:09:31.39+07
10	patient2@email.com	$2a$10$jO/gTShr6wTmVIwkZjsyjuvneY1BOKQMd9y99Ipd4eOdS0pSLfXVe	Patient Two	\N	\N	\N	t	4	2025-11-16 17:09:31.449+07	2025-11-16 17:09:31.449+07
11	patient3@email.com	$2a$10$9xP63gOGU7HYDF5ldPDS.OCgtMH4e6411HJSjkWY5TbVqnuZLZX6C	Patient Three	\N	\N	\N	t	4	2025-11-16 17:09:31.507+07	2025-11-16 17:09:31.507+07
12	patient4@email.com	$2a$10$58hs1.JignptbxXB4AJ1COJ.hL8v3IsYoN8voTtVbycCGSA4qazlG	Patient Four	\N	\N	\N	t	4	2025-11-16 17:09:31.565+07	2025-11-16 17:09:31.565+07
7	dr3@email.com	$2a$10$W1AbeEYkNEsHFws8uYoQFeKL0NDXwmrm6NdSu3GT9f6SMYJFsa3Fq	Dr. Le Van C	0785245277	\N	\N	t	2	2025-11-16 17:09:31.273+07	2025-11-16 17:09:31.273+07
2	doctor@email.com	$2a$10$Oee4aTckdsHTejjCI5O/te.clEQSJlo28WqIfJjvzPwUZG1fyVdX2	Doctor Account	0123123123	2025-11-02	\N	t	2	2025-11-16 17:09:30.977+07	2025-11-16 17:09:30.977+07
4	patient@email.com	$2a$10$lhiaLck1licKfIWYRld.Iegj5q8cuhU8yLKHP4gll5270oaihlrq2	Patient Account111	0785235277	2025-11-05	\N	t	4	2025-11-16 17:09:31.094+07	2025-11-16 17:09:31.094+07
27	tes111t@email.com	$2a$10$rYAiwVPv9IEawFOqlGxHW.Czq0YEr1IWpKXA.P1yNp3PB3V.MRrgG	tu	0123456789	2025-11-11	\N	t	4	2025-11-18 16:04:41.718+07	2025-11-18 16:04:41.718+07
28	test1121311@email.com	$2a$10$PNXXA8Nrg9ANaebGmYKueeGgm0vho969tI3IB9TuLzTjFSxiaVl8u	áđá	0213123123	2025-11-05	\N	t	4	2025-11-18 16:37:57.366+07	2025-11-18 16:37:57.366+07
29	patient+1763458882835@example.test	$2a$10$hoSnNc86Vdz5Go0wkUBPNOuTOf1y.VQQWJnfeX7.X3pGwUVzx//JC	ádsad	0123123123	2025-11-12	\N	t	4	2025-11-18 16:41:22.961+07	2025-11-18 16:41:22.961+07
30	tes1121t@email.com	$2a$10$SSLXXkptkNiqr0L7EOMTeOqw1gy8zvpe8DY5UtKRz8WEug8e2.FeK	àqưeãzc	0987654321	2025-11-11	\N	t	4	2025-11-18 17:07:20.87+07	2025-11-18 17:07:20.87+07
14	admin@example.com	$2a$10$rjy6G6CwULN.W4PC9Vd19eaKdK2UqtAFuGLuu8crPf2k86Rt.Hj6W	Administrator	\N	\N	\N	t	1	2025-11-17 19:41:28.04+07	2025-11-17 19:41:28.04+07
15	doctor@example.com	$2a$10$SkF2eMm38fm3FmfelMuv1ezMBJY/ITijYE3PhxXYLwkq.BmNFo.wm	Doctor Account	\N	\N	\N	t	2	2025-11-17 19:41:28.359+07	2025-11-17 19:41:28.359+07
16	reception@example.com	$2a$10$7/UegeuzcVhPfpwL7qaHheTwUXziU6/LDOwd66pu1xYrsmBWPmaJm	Reception Account	\N	\N	\N	t	3	2025-11-17 19:41:28.76+07	2025-11-17 19:41:28.76+07
17	patient@example.com	$2a$10$vPOgFFQyWi07NpcDWWN14e0n6qkhz3MVAUBrIymiHibMvB5C4sSCy	Patient Account	\N	\N	\N	t	4	2025-11-17 19:41:29.026+07	2025-11-17 19:41:29.026+07
31	honma953@gmail.com	$2a$10$4JmknNyleqyQfyefkV/H0uDeMJTOQzeuPvQW1m.R04c8lGS509RJy	moi	0786542323	2025-11-05	\N	t	4	2025-11-18 17:24:50.567+07	2025-11-18 17:24:50.567+07
32	pharmacist@email.com	$2a$06$FqQ.C/g9I6OCfI22hCI/TOCcZAnzs0enh.nj0p1PQBxozTWWy3rNa	Dược sĩ chính	0133456789	\N	\N	t	6	2025-11-20 03:57:33.536886+07	2025-11-20 03:57:33.536886+07
18	dr1@example.test	$2a$10$7MdZdH9zsV.Vvm5xjQxO7.uKqHH1REHJ3ITYPPN730/2QQ2YhKuoq	Dr. Nguyen Van A	\N	\N	\N	t	2	2025-11-17 19:41:29.307+07	2025-11-17 19:41:29.307+07
19	dr2@example.test	$2a$10$ONCmnjdFmMCzWz50Y17Cf.nbCxOAMnbxSjj5EfjAGabP93PA07Dp6	Dr. Tran Thi B	\N	\N	\N	t	2	2025-11-17 19:42:37.063+07	2025-11-17 19:42:37.063+07
20	dr3@example.test	$2a$10$/A2pLoS9QxYfuVlYlKlG/esdR.QoTmVnvLDxrDWqfnP.HMaFDSd.W	Dr. Le Van C	\N	\N	\N	t	2	2025-11-17 19:42:37.392+07	2025-11-17 19:42:37.392+07
21	dr4@example.test	$2a$10$ktt6tORa6rURz5TbfpumWOYwIR.NDlUEa8xEPFbbXB7iysxe/9Uuy	Dr. Pham Thi D	\N	\N	\N	t	2	2025-11-17 19:42:37.779+07	2025-11-17 19:42:37.779+07
22	patient1@example.test	$2a$10$9zM5b0I9apoUGZVKdZYPtesRoKhQ2Bk5OGtUxR2FwWc9UoANTFJxa	Patient One	\N	\N	\N	t	4	2025-11-17 19:42:38.056+07	2025-11-17 19:42:38.056+07
23	patient2@example.test	$2a$10$O17Dv07EXvER5W8X0XKdzehsIAlGAC9yF56bZy53QXN7ujVPxtZ9.	Patient Two	\N	\N	\N	t	4	2025-11-17 19:42:38.39+07	2025-11-17 19:42:38.39+07
24	patient3@example.test	$2a$10$JNlUloYuh8LqVGPPaucYKenp4Urjsd92Pm9AWlUc.qpg21QTjsh8u	Patient Three	\N	\N	\N	t	4	2025-11-17 19:42:38.686+07	2025-11-17 19:42:38.686+07
25	patient4@example.test	$2a$10$EbShW6QrdHZ85MCxwDDPw.C42Rab3rNwW6jPPZ9g/5oYEZhgyeCkm	Patient Four	\N	\N	\N	t	4	2025-11-17 19:42:39.005+07	2025-11-17 19:42:39.005+07
26	patient+1763386443076@example.test	$2a$10$yFJmyY1C/6M/GWcT4AEAcu8Pj0lgU8/7VKamvQiUIOmE14Y8AjjBe	aaa	0123123123	2025-11-12	\N	t	4	2025-11-17 20:34:03.492+07	2025-11-17 20:34:03.492+07
\.


--
-- TOC entry 5274 (class 0 OID 0)
-- Dependencies: 237
-- Name: appointments_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.appointments_id_seq', 55, true);


--
-- TOC entry 5275 (class 0 OID 0)
-- Dependencies: 255
-- Name: audit_logs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.audit_logs_id_seq', 20, true);


--
-- TOC entry 5276 (class 0 OID 0)
-- Dependencies: 257
-- Name: doctor_reviews_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.doctor_reviews_id_seq', 1, false);


--
-- TOC entry 5277 (class 0 OID 0)
-- Dependencies: 227
-- Name: doctor_specialties_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.doctor_specialties_id_seq', 8, true);


--
-- TOC entry 5278 (class 0 OID 0)
-- Dependencies: 223
-- Name: doctors_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.doctors_id_seq', 9, true);


--
-- TOC entry 5279 (class 0 OID 0)
-- Dependencies: 249
-- Name: invoices_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.invoices_id_seq', 29, true);


--
-- TOC entry 5280 (class 0 OID 0)
-- Dependencies: 251
-- Name: lab_orders_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.lab_orders_id_seq', 10, true);


--
-- TOC entry 5281 (class 0 OID 0)
-- Dependencies: 239
-- Name: medical_records_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.medical_records_id_seq', 10, true);


--
-- TOC entry 5282 (class 0 OID 0)
-- Dependencies: 241
-- Name: medicines_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.medicines_id_seq', 5, true);


--
-- TOC entry 5283 (class 0 OID 0)
-- Dependencies: 253
-- Name: notifications_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.notifications_id_seq', 33, true);


--
-- TOC entry 5284 (class 0 OID 0)
-- Dependencies: 229
-- Name: patients_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.patients_id_seq', 30, true);


--
-- TOC entry 5285 (class 0 OID 0)
-- Dependencies: 245
-- Name: prescriptions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.prescriptions_id_seq', 28, true);


--
-- TOC entry 5286 (class 0 OID 0)
-- Dependencies: 219
-- Name: roles_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.roles_id_seq', 6, true);


--
-- TOC entry 5287 (class 0 OID 0)
-- Dependencies: 231
-- Name: rooms_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.rooms_id_seq', 12, true);


--
-- TOC entry 5288 (class 0 OID 0)
-- Dependencies: 233
-- Name: schedules_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.schedules_id_seq', 128, true);


--
-- TOC entry 5289 (class 0 OID 0)
-- Dependencies: 225
-- Name: specialties_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.specialties_id_seq', 5, true);


--
-- TOC entry 5290 (class 0 OID 0)
-- Dependencies: 243
-- Name: stocks_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.stocks_id_seq', 5, true);


--
-- TOC entry 5291 (class 0 OID 0)
-- Dependencies: 247
-- Name: suppliers_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.suppliers_id_seq', 3, true);


--
-- TOC entry 5292 (class 0 OID 0)
-- Dependencies: 235
-- Name: timeslots_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.timeslots_id_seq', 1097, true);


--
-- TOC entry 5293 (class 0 OID 0)
-- Dependencies: 221
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.users_id_seq', 32, true);


--
-- TOC entry 4964 (class 2606 OID 31215)
-- Name: _prisma_migrations _prisma_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public._prisma_migrations
    ADD CONSTRAINT _prisma_migrations_pkey PRIMARY KEY (id);


--
-- TOC entry 4999 (class 2606 OID 31354)
-- Name: appointments appointments_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.appointments
    ADD CONSTRAINT appointments_pkey PRIMARY KEY (id);


--
-- TOC entry 5025 (class 2606 OID 31459)
-- Name: audit_logs audit_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.audit_logs
    ADD CONSTRAINT audit_logs_pkey PRIMARY KEY (id);


--
-- TOC entry 5027 (class 2606 OID 38302)
-- Name: doctor_reviews doctor_reviews_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctor_reviews
    ADD CONSTRAINT doctor_reviews_pkey PRIMARY KEY (id);


--
-- TOC entry 4980 (class 2606 OID 31299)
-- Name: doctor_specialties doctor_specialties_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctor_specialties
    ADD CONSTRAINT doctor_specialties_pkey PRIMARY KEY (id);


--
-- TOC entry 4972 (class 2606 OID 31280)
-- Name: doctors doctors_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctors
    ADD CONSTRAINT doctors_pkey PRIMARY KEY (id);


--
-- TOC entry 5019 (class 2606 OID 31426)
-- Name: invoices invoices_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoices
    ADD CONSTRAINT invoices_pkey PRIMARY KEY (id);


--
-- TOC entry 5021 (class 2606 OID 31438)
-- Name: lab_orders lab_orders_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lab_orders
    ADD CONSTRAINT lab_orders_pkey PRIMARY KEY (id);


--
-- TOC entry 5005 (class 2606 OID 31365)
-- Name: medical_records medical_records_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.medical_records
    ADD CONSTRAINT medical_records_pkey PRIMARY KEY (id);


--
-- TOC entry 5008 (class 2606 OID 31377)
-- Name: medicines medicines_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.medicines
    ADD CONSTRAINT medicines_pkey PRIMARY KEY (id);


--
-- TOC entry 5023 (class 2606 OID 31449)
-- Name: notifications notifications_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.notifications
    ADD CONSTRAINT notifications_pkey PRIMARY KEY (id);


--
-- TOC entry 4988 (class 2606 OID 31310)
-- Name: patients patients_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.patients
    ADD CONSTRAINT patients_pkey PRIMARY KEY (id);


--
-- TOC entry 5014 (class 2606 OID 31400)
-- Name: prescriptions prescriptions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prescriptions
    ADD CONSTRAINT prescriptions_pkey PRIMARY KEY (id);


--
-- TOC entry 4967 (class 2606 OID 31256)
-- Name: roles roles_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.roles
    ADD CONSTRAINT roles_pkey PRIMARY KEY (id);


--
-- TOC entry 4990 (class 2606 OID 31321)
-- Name: rooms rooms_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rooms
    ADD CONSTRAINT rooms_pkey PRIMARY KEY (id);


--
-- TOC entry 4992 (class 2606 OID 31329)
-- Name: schedules schedules_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.schedules
    ADD CONSTRAINT schedules_pkey PRIMARY KEY (id);


--
-- TOC entry 4976 (class 2606 OID 31291)
-- Name: specialties specialties_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.specialties
    ADD CONSTRAINT specialties_pkey PRIMARY KEY (id);


--
-- TOC entry 5011 (class 2606 OID 31387)
-- Name: stocks stocks_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stocks
    ADD CONSTRAINT stocks_pkey PRIMARY KEY (id);


--
-- TOC entry 5016 (class 2606 OID 31410)
-- Name: suppliers suppliers_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.suppliers
    ADD CONSTRAINT suppliers_pkey PRIMARY KEY (id);


--
-- TOC entry 4997 (class 2606 OID 31341)
-- Name: timeslots timeslots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.timeslots
    ADD CONSTRAINT timeslots_pkey PRIMARY KEY (id);


--
-- TOC entry 4970 (class 2606 OID 31268)
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- TOC entry 4978 (class 1259 OID 31467)
-- Name: doctor_specialties_doctor_id_specialty_id_key; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX doctor_specialties_doctor_id_specialty_id_key ON public.doctor_specialties USING btree (doctor_id, specialty_id);


--
-- TOC entry 4973 (class 1259 OID 31462)
-- Name: doctors_user_id_key; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX doctors_user_id_key ON public.doctors USING btree (user_id);


--
-- TOC entry 5000 (class 1259 OID 31471)
-- Name: idx_appointments_doctor_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_appointments_doctor_date ON public.appointments USING btree (doctor_id, appointment_date);


--
-- TOC entry 5001 (class 1259 OID 31472)
-- Name: idx_appointments_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_appointments_status ON public.appointments USING btree (status);


--
-- TOC entry 4981 (class 1259 OID 31466)
-- Name: idx_doctorspecialty_doctor; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_doctorspecialty_doctor ON public.doctor_specialties USING btree (doctor_id);


--
-- TOC entry 4982 (class 1259 OID 31465)
-- Name: idx_doctorspecialty_specialty; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_doctorspecialty_specialty ON public.doctor_specialties USING btree (specialty_id);


--
-- TOC entry 5017 (class 1259 OID 31478)
-- Name: idx_invoices_patient; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_invoices_patient ON public.invoices USING btree (patient_id);


--
-- TOC entry 5003 (class 1259 OID 31474)
-- Name: idx_medicalrecords_patient; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_medicalrecords_patient ON public.medical_records USING btree (patient_id);


--
-- TOC entry 4983 (class 1259 OID 39981)
-- Name: idx_patients_id_number; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_patients_id_number ON public.patients USING btree (id_number);


--
-- TOC entry 4984 (class 1259 OID 39979)
-- Name: idx_patients_new_province; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_patients_new_province ON public.patients USING btree (new_province);


--
-- TOC entry 4985 (class 1259 OID 39978)
-- Name: idx_patients_old_province; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_patients_old_province ON public.patients USING btree (old_province);


--
-- TOC entry 4986 (class 1259 OID 39149)
-- Name: idx_patients_owner_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_patients_owner_user_id ON public.patients USING btree (owner_user_id);


--
-- TOC entry 5012 (class 1259 OID 31477)
-- Name: idx_prescriptions_patient; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_prescriptions_patient ON public.prescriptions USING btree (patient_id);


--
-- TOC entry 5028 (class 1259 OID 38304)
-- Name: idx_reviews_doctor; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_reviews_doctor ON public.doctor_reviews USING btree (doctor_id);


--
-- TOC entry 5029 (class 1259 OID 38305)
-- Name: idx_reviews_patient; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_reviews_patient ON public.doctor_reviews USING btree (patient_id);


--
-- TOC entry 4993 (class 1259 OID 67573)
-- Name: idx_timeslots_active; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_timeslots_active ON public.timeslots USING btree (is_active);


--
-- TOC entry 4994 (class 1259 OID 31469)
-- Name: idx_timeslots_doctor_date; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_timeslots_doctor_date ON public.timeslots USING btree (doctor_id, date);


--
-- TOC entry 5006 (class 1259 OID 31475)
-- Name: medicines_code_key; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX medicines_code_key ON public.medicines USING btree (code);


--
-- TOC entry 4965 (class 1259 OID 31460)
-- Name: roles_name_key; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX roles_name_key ON public.roles USING btree (name);


--
-- TOC entry 4974 (class 1259 OID 31463)
-- Name: specialties_name_key; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX specialties_name_key ON public.specialties USING btree (name);


--
-- TOC entry 4977 (class 1259 OID 31464)
-- Name: specialties_slug_key; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX specialties_slug_key ON public.specialties USING btree (slug);


--
-- TOC entry 5009 (class 1259 OID 31476)
-- Name: stocks_medicine_id_batch_number_key; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX stocks_medicine_id_batch_number_key ON public.stocks USING btree (medicine_id, batch_number);


--
-- TOC entry 4995 (class 1259 OID 31470)
-- Name: timeslot_unique; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX timeslot_unique ON public.timeslots USING btree (doctor_id, date, start_time, end_time);


--
-- TOC entry 5002 (class 1259 OID 31473)
-- Name: uq_appointments_patient_timeslot; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX uq_appointments_patient_timeslot ON public.appointments USING btree (patient_id, timeslot_id);


--
-- TOC entry 5030 (class 1259 OID 38303)
-- Name: uq_doctor_patient_review; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX uq_doctor_patient_review ON public.doctor_reviews USING btree (doctor_id, patient_id);


--
-- TOC entry 4968 (class 1259 OID 31461)
-- Name: users_email_key; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX users_email_key ON public.users USING btree (email);


--
-- TOC entry 5040 (class 2606 OID 31519)
-- Name: appointments appointments_created_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.appointments
    ADD CONSTRAINT appointments_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- TOC entry 5041 (class 2606 OID 31524)
-- Name: appointments appointments_doctor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.appointments
    ADD CONSTRAINT appointments_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctors(id) ON DELETE CASCADE;


--
-- TOC entry 5042 (class 2606 OID 31529)
-- Name: appointments appointments_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.appointments
    ADD CONSTRAINT appointments_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patients(id) ON DELETE CASCADE;


--
-- TOC entry 5043 (class 2606 OID 31534)
-- Name: appointments appointments_timeslot_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.appointments
    ADD CONSTRAINT appointments_timeslot_id_fkey FOREIGN KEY (timeslot_id) REFERENCES public.timeslots(id) ON DELETE SET NULL;


--
-- TOC entry 5058 (class 2606 OID 38306)
-- Name: doctor_reviews doctor_reviews_doctor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctor_reviews
    ADD CONSTRAINT doctor_reviews_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctors(id) ON DELETE CASCADE;


--
-- TOC entry 5059 (class 2606 OID 38311)
-- Name: doctor_reviews doctor_reviews_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctor_reviews
    ADD CONSTRAINT doctor_reviews_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patients(id) ON DELETE CASCADE;


--
-- TOC entry 5033 (class 2606 OID 31489)
-- Name: doctor_specialties doctor_specialties_doctor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctor_specialties
    ADD CONSTRAINT doctor_specialties_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctors(id) ON DELETE CASCADE;


--
-- TOC entry 5034 (class 2606 OID 31494)
-- Name: doctor_specialties doctor_specialties_specialty_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctor_specialties
    ADD CONSTRAINT doctor_specialties_specialty_id_fkey FOREIGN KEY (specialty_id) REFERENCES public.specialties(id) ON DELETE CASCADE;


--
-- TOC entry 5032 (class 2606 OID 31484)
-- Name: doctors doctors_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.doctors
    ADD CONSTRAINT doctors_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- TOC entry 5035 (class 2606 OID 39144)
-- Name: patients fk_patients_owner_user_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.patients
    ADD CONSTRAINT fk_patients_owner_user_id FOREIGN KEY (owner_user_id) REFERENCES public.users(id);


--
-- TOC entry 5051 (class 2606 OID 31574)
-- Name: invoices invoices_appointment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoices
    ADD CONSTRAINT invoices_appointment_id_fkey FOREIGN KEY (appointment_id) REFERENCES public.appointments(id) ON DELETE SET NULL;


--
-- TOC entry 5052 (class 2606 OID 31579)
-- Name: invoices invoices_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoices
    ADD CONSTRAINT invoices_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patients(id) ON DELETE CASCADE;


--
-- TOC entry 5053 (class 2606 OID 40830)
-- Name: invoices invoices_prescription_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoices
    ADD CONSTRAINT invoices_prescription_id_fkey FOREIGN KEY (prescription_id) REFERENCES public.prescriptions(id) ON DELETE SET NULL;


--
-- TOC entry 5054 (class 2606 OID 31584)
-- Name: lab_orders lab_orders_appointment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lab_orders
    ADD CONSTRAINT lab_orders_appointment_id_fkey FOREIGN KEY (appointment_id) REFERENCES public.appointments(id) ON DELETE SET NULL;


--
-- TOC entry 5055 (class 2606 OID 31589)
-- Name: lab_orders lab_orders_doctor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lab_orders
    ADD CONSTRAINT lab_orders_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctors(id) ON DELETE SET NULL;


--
-- TOC entry 5056 (class 2606 OID 31594)
-- Name: lab_orders lab_orders_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lab_orders
    ADD CONSTRAINT lab_orders_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patients(id) ON DELETE CASCADE;


--
-- TOC entry 5044 (class 2606 OID 31539)
-- Name: medical_records medical_records_appointment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.medical_records
    ADD CONSTRAINT medical_records_appointment_id_fkey FOREIGN KEY (appointment_id) REFERENCES public.appointments(id) ON DELETE SET NULL;


--
-- TOC entry 5045 (class 2606 OID 31544)
-- Name: medical_records medical_records_doctor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.medical_records
    ADD CONSTRAINT medical_records_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctors(id) ON DELETE SET NULL;


--
-- TOC entry 5046 (class 2606 OID 31549)
-- Name: medical_records medical_records_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.medical_records
    ADD CONSTRAINT medical_records_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patients(id) ON DELETE CASCADE;


--
-- TOC entry 5057 (class 2606 OID 31599)
-- Name: notifications notifications_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.notifications
    ADD CONSTRAINT notifications_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- TOC entry 5036 (class 2606 OID 31499)
-- Name: patients patients_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.patients
    ADD CONSTRAINT patients_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- TOC entry 5048 (class 2606 OID 31559)
-- Name: prescriptions prescriptions_appointment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prescriptions
    ADD CONSTRAINT prescriptions_appointment_id_fkey FOREIGN KEY (appointment_id) REFERENCES public.appointments(id) ON DELETE SET NULL;


--
-- TOC entry 5049 (class 2606 OID 31564)
-- Name: prescriptions prescriptions_doctor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prescriptions
    ADD CONSTRAINT prescriptions_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctors(id) ON DELETE SET NULL;


--
-- TOC entry 5050 (class 2606 OID 31569)
-- Name: prescriptions prescriptions_patient_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prescriptions
    ADD CONSTRAINT prescriptions_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patients(id) ON DELETE CASCADE;


--
-- TOC entry 5037 (class 2606 OID 31504)
-- Name: schedules schedules_doctor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.schedules
    ADD CONSTRAINT schedules_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctors(id) ON DELETE CASCADE;


--
-- TOC entry 5038 (class 2606 OID 31509)
-- Name: schedules schedules_room_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.schedules
    ADD CONSTRAINT schedules_room_id_fkey FOREIGN KEY (room_id) REFERENCES public.rooms(id) ON DELETE SET NULL;


--
-- TOC entry 5047 (class 2606 OID 31554)
-- Name: stocks stocks_medicine_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stocks
    ADD CONSTRAINT stocks_medicine_id_fkey FOREIGN KEY (medicine_id) REFERENCES public.medicines(id) ON DELETE CASCADE;


--
-- TOC entry 5039 (class 2606 OID 31514)
-- Name: timeslots timeslots_doctor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.timeslots
    ADD CONSTRAINT timeslots_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctors(id) ON DELETE CASCADE;


--
-- TOC entry 5031 (class 2606 OID 31479)
-- Name: users users_role_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(id) ON DELETE RESTRICT;


--
-- TOC entry 5252 (class 0 OID 0)
-- Dependencies: 6
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;


-- Completed on 2025-11-22 00:15:17

--
-- PostgreSQL database dump complete
--

