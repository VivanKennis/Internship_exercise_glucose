#include "mathaddon.h"
#include <Models_M_API.h>
#include "sund_sundials_flags.h"

#include <Python.h>

#include <cmath>
#include <vector>

using namespace std;


static ModelFunction M_glucose7_modelfunction;
static ICFunction M_glucose7_initialcondition;
static PyObject *M_glucose7_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

// states, features, outputs, inputs, events, parameters
const int M_glucose7_numberof[6] = {
	18,23,0,9,1,59};
const char *M_glucose7_statenames[18] = {
	"PVo2max","O2_deficit","lactate","met_stress","exercise_drive","NOR_neuronal","NOR_adrenal","Epi_adrenal","NOR","EPI","I","X","G","Gprod","Gup","Ie","Ggly","A"};
const char *M_glucose7_featurenames[23] = {
	"Glucose_mmolL","Insulin_mUL","Epinephrine_nmolL","Norepinephrine_nmolL","Lactate_mmolL","PVo2max","O2_deficit","lactate","met_stress","exercise_drive","NOR_neuronal","NOR_adrenal","Epi_adrenal","EPI","NOR","X","I","G","Gprod","Gup",
	"Ie","Ggly","A"};
const char *M_glucose7_featureunits[23] = {
	"1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
	"1","1","1"};
const char *M_glucose7_outputnames[1]{};
const char *M_glucose7_inputnames[9] = {
	"intensity","Ib","Gb","BW","Epib","Norb","Lacb","time_ex","Vo2max"};
const char *M_glucose7_parameternames[59] = {
	"vmax_thresh","km_thresh","vmax_o2","k_O2","vmax_lac","km_lac","n_lac","elim_lactate","k_stress","elim_stress","lac_stress","k_push","k_push2","k_rec_drive","Vmax_stres_n","km_stres_n","Vmax_ex_n","km_ex_n","spill","elim_NOR_neuronal",
	"Vmax_stres_a","km_stres_a","Vmax_ex_a","km_ex_a","n_ex_a","scale","conv","release_epi","elim_nor_plasma","elim_epi_plasma","k_I_stim","elim_insu","k_I_epi","k_I_nor","Vmax_I_epi","Vmax_I_nor","km_I_epi","km_I_nor","n_I_epi","n_I_nor",
	"elim_X","ins_action","Vmax_X_epi","km_X_epi","elim_glu","k_Gpro_stim","k_Gpro_decay","Vmax_Gprod_epi","Vmax_Gprod_nor","km_Gprod_epi","km_Gprod_nor","n_Gprod_epi","n_Gprod_nor","K_Gup_stim","k_Gup_decay","K_Ie_stim","k_Ie_decay","k","T1"};
const char *M_glucose7_eventnames[1] = {
	"event_basal"};

// {0 = mandatory, 1 = non-mandatory, 2 = non-mandatory - zero default value}
const int M_glucose7_mandatoryinputs[9] = {
	2,1,1,1,1,1,1,2,1};

const std::vector<double> M_glucose7_defaultInputs{0.0, 12.0, 4.9750, 84.0, 0.3, 1.3, 1.0, 0.0, 40.0};
// {#nrdifferentialstates,<differentialstate index>}
const int M_glucose7_differentialstates[19] = {
	18,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};
// {#depency,<input numbers>}
const int M_glucose7_inputdependency[1] = {
	0};
const double M_glucose7_defaultparameters[59] = {
	100.0,0.68,0.94931108961551,13.980781172760825,399.0018997915273,6.62732270079653,6.433343189248456,0.036958560209759,0.004387879149666,9.989663354499742,0.025869444770412,0.00095170670819,0.19180708426934,0.570149831504022,15.088214310775609,224.33188192026978,5.074512306315921,0.010425416911303,0.468094380964438,2.393627007141584,
	11.48471338138073,0.268851356744835,163.9183586402966,68.44518510913419,3.999998458440444,0.114655263975009,0.999999999474022,0.444241773803292,0.456405829227235,9.999994028193964,0.1,0.142,0.1,0.1,0.1,0.1,1.0,2.0,1.0,1.0,
	0.05,0.000028,1.0,0.1,0.035,0.000008769982238011767,0.056,10.0,10.0,1.0,2.0,1.0,1.0,0.000010823712255772646,0.0485,0.00125,0.075,0.0000599467,6.0};
const char M_glucose7_timeunit[] = "m";

static double f_hill(double x,double  km,double  n){
	return pow(x,n) / (pow(km,n) + pow(x,n));
}

static double f_mm(double x,double  km){
	return x / (km + x);
}

static double f_2(double x,double  a){
	return a * (pow(x,2.0));
}

static void M_glucose7_modelfunction(double time, double timescale, double *statevector, double *derivativevector, double *RESvector, double *parametervector, double *featurevector, double *outputvector, double **inputvector, double *eventvector, int *eventstatus, int DOflag){
	double PVo2max,O2_deficit,lactate,met_stress,exercise_drive,NOR_neuronal,NOR_adrenal,Epi_adrenal,NOR,EPI,I,X,G,Gprod,Gup,Ie,Ggly,A;
	double vmax_thresh,km_thresh,vmax_o2,k_O2,vmax_lac,km_lac,n_lac,elim_lactate,k_stress,elim_stress,lac_stress,k_push,k_push2,k_rec_drive,Vmax_stres_n,km_stres_n,Vmax_ex_n,km_ex_n,spill,elim_NOR_neuronal;
	double Vmax_stres_a,km_stres_a,Vmax_ex_a,km_ex_a,n_ex_a,scale,conv,release_epi,elim_nor_plasma,elim_epi_plasma,k_I_stim,elim_insu,k_I_epi,k_I_nor,Vmax_I_epi,Vmax_I_nor,km_I_epi,km_I_nor,n_I_epi,n_I_nor;
	double elim_X,ins_action,Vmax_X_epi,km_X_epi,elim_glu,k_Gpro_stim,k_Gpro_decay,Vmax_Gprod_epi,Vmax_Gprod_nor,km_Gprod_epi,km_Gprod_nor,n_Gprod_epi,n_Gprod_nor,K_Gup_stim,k_Gup_decay,K_Ie_stim,k_Ie_decay,k,T1;
	[[maybe_unused]] double u3,Ib,Gb,BW,Epib,Norb,Lacb,time_ex,Vo2max;
	double release_nor_adrenal,conv_nor_to_epi,Vmax_stres_a_eff,Vmax_ex_a_eff,threshold,ATH,gly_A,A_u3,B_u3,VolG,I_epi_nor,epi_nor;

	PVo2max = statevector[0];
	O2_deficit = statevector[1];
	lactate = statevector[2];
	met_stress = statevector[3];
	exercise_drive = statevector[4];
	NOR_neuronal = statevector[5];
	NOR_adrenal = statevector[6];
	Epi_adrenal = statevector[7];
	NOR = statevector[8];
	EPI = statevector[9];
	I = statevector[10];
	X = statevector[11];
	G = statevector[12];
	Gprod = statevector[13];
	Gup = statevector[14];
	Ie = statevector[15];
	Ggly = statevector[16];
	A = statevector[17];
	vmax_thresh = parametervector[0];
	km_thresh = parametervector[1];
	vmax_o2 = parametervector[2];
	k_O2 = parametervector[3];
	vmax_lac = parametervector[4];
	km_lac = parametervector[5];
	n_lac = parametervector[6];
	elim_lactate = parametervector[7];
	k_stress = parametervector[8];
	elim_stress = parametervector[9];
	lac_stress = parametervector[10];
	k_push = parametervector[11];
	k_push2 = parametervector[12];
	k_rec_drive = parametervector[13];
	Vmax_stres_n = parametervector[14];
	km_stres_n = parametervector[15];
	Vmax_ex_n = parametervector[16];
	km_ex_n = parametervector[17];
	spill = parametervector[18];
	elim_NOR_neuronal = parametervector[19];
	Vmax_stres_a = parametervector[20];
	km_stres_a = parametervector[21];
	Vmax_ex_a = parametervector[22];
	km_ex_a = parametervector[23];
	n_ex_a = parametervector[24];
	scale = parametervector[25];
	conv = parametervector[26];
	release_epi = parametervector[27];
	elim_nor_plasma = parametervector[28];
	elim_epi_plasma = parametervector[29];
	k_I_stim = parametervector[30];
	elim_insu = parametervector[31];
	k_I_epi = parametervector[32];
	k_I_nor = parametervector[33];
	Vmax_I_epi = parametervector[34];
	Vmax_I_nor = parametervector[35];
	km_I_epi = parametervector[36];
	km_I_nor = parametervector[37];
	n_I_epi = parametervector[38];
	n_I_nor = parametervector[39];
	elim_X = parametervector[40];
	ins_action = parametervector[41];
	Vmax_X_epi = parametervector[42];
	km_X_epi = parametervector[43];
	elim_glu = parametervector[44];
	k_Gpro_stim = parametervector[45];
	k_Gpro_decay = parametervector[46];
	Vmax_Gprod_epi = parametervector[47];
	Vmax_Gprod_nor = parametervector[48];
	km_Gprod_epi = parametervector[49];
	km_Gprod_nor = parametervector[50];
	n_Gprod_epi = parametervector[51];
	n_Gprod_nor = parametervector[52];
	K_Gup_stim = parametervector[53];
	k_Gup_decay = parametervector[54];
	K_Ie_stim = parametervector[55];
	k_Ie_decay = parametervector[56];
	k = parametervector[57];
	T1 = parametervector[58];
	u3 = (inputvector[0]) ? (*inputvector[0]) : 0.0;
	Ib = (inputvector[1]) ? (*inputvector[1]) : 12.0;
	Gb = (inputvector[2]) ? (*inputvector[2]) : 4.9750;
	BW = (inputvector[3]) ? (*inputvector[3]) : 84.0;
	Epib = (inputvector[4]) ? (*inputvector[4]) : 0.3;
	Norb = (inputvector[5]) ? (*inputvector[5]) : 1.3;
	Lacb = (inputvector[6]) ? (*inputvector[6]) : 1.0;
	time_ex = (inputvector[7]) ? (*inputvector[7]) : 0.0;
	Vo2max = (inputvector[8]) ? (*inputvector[8]) : 40.0;
	release_nor_adrenal = conv*0.2;
	conv_nor_to_epi = conv*0.8;
	Vmax_stres_a_eff = Vmax_stres_a * (1.0+ scale * max(Vo2max-40.0,0.0));
	Vmax_ex_a_eff = Vmax_ex_a * (1.0+ scale * max(Vo2max-40.0,0.0));
	threshold = vmax_thresh * f_mm(Vo2max/90.0, km_thresh);
	ATH = (-1.1521*pow(PVo2max,2.0))+(87.471*PVo2max);
	gly_A = (sign(A-ATH+1.0)+1.0)/2.0;
	A_u3 = 1.0*sign(PVo2max);
	B_u3 = (sign(1.0-PVo2max)+1.0)/2.0;
	VolG = 11.7;
	I_epi_nor = Vmax_I_epi * f_hill(max((EPI-Epib),0.0), km_I_epi, n_I_epi) + Vmax_I_nor * f_hill(max((NOR-Norb),0.0), km_I_nor, n_I_nor);
	epi_nor = Vmax_Gprod_epi * f_hill(max((EPI-Epib),0.0), km_Gprod_epi, n_Gprod_epi) + Vmax_Gprod_nor * f_hill(max((NOR-Norb),0.0), km_Gprod_nor, n_Gprod_nor);

	if (DOflag == DOFLAG_DDT) {
		derivativevector[0] = timescale * (-0.8*PVo2max + 0.8*u3);
		derivativevector[1] = timescale * (max(vmax_o2 * max(PVo2max-threshold,0.0),0.0) - k_O2 * O2_deficit);
		derivativevector[2] = timescale * (vmax_lac * f_hill(max(O2_deficit,0.0), km_lac, n_lac) - elim_lactate*(lactate-Lacb));
		derivativevector[3] = timescale * (f_2(max(PVo2max-threshold,0.0), k_stress) + lac_stress * max(lactate-Lacb,0.0) - elim_stress*met_stress);
		derivativevector[4] = timescale * ((k_push *  max(time - time_ex, 0.0) + k_push2) * PVo2max* max(sign(u3),0.0) - k_rec_drive * exercise_drive);
		derivativevector[5] = timescale * (Vmax_stres_n * max(met_stress,0.0) + Vmax_ex_n * f_mm(max(exercise_drive,0.0), km_ex_n) - spill * NOR_neuronal - elim_NOR_neuronal * NOR_neuronal);
		derivativevector[6] = timescale * (Vmax_stres_a_eff * f_mm(max(met_stress,0.0), km_stres_a) + Vmax_ex_a_eff * f_hill(max(exercise_drive,0.0), km_ex_a, n_ex_a) - release_nor_adrenal * NOR_adrenal - conv_nor_to_epi * NOR_adrenal);
		derivativevector[7] = timescale * (conv_nor_to_epi * NOR_adrenal - release_epi * Epi_adrenal);
		derivativevector[8] = timescale * (spill * NOR_neuronal + release_nor_adrenal * NOR_adrenal- elim_nor_plasma*(NOR-Norb));
		derivativevector[9] = timescale * (release_epi * Epi_adrenal - elim_epi_plasma * (EPI-Epib));
		derivativevector[10] = timescale * (k_I_stim*max(G-Gb,0.0)/(1.0+k_I_epi*max(EPI-Epib,0.0) + k_I_nor*max((NOR-Norb),0.0)) - elim_insu * (I- Ib) - Ie);
		derivativevector[11] = timescale * (ins_action *(I - Ib) - elim_X * X - Vmax_X_epi * f_mm(max((EPI-Epib),0.0), km_X_epi) * X);
		derivativevector[12] = timescale * ((BW/VolG)*(Gprod - Ggly) - (BW/VolG) * Gup - X*G - elim_glu *(G - Gb));
		derivativevector[13] = timescale * (k_Gpro_stim * PVo2max - k_Gpro_decay * Gprod + epi_nor);
		derivativevector[14] = timescale * (K_Gup_stim  * PVo2max - k_Gup_decay * Gup);
		derivativevector[15] = timescale * (K_Ie_stim * PVo2max  - k_Ie_decay * Ie);
		derivativevector[16] = timescale * (gly_A*k*A_u3 - Ggly/T1*B_u3);
		derivativevector[17] = timescale * (A_u3*PVo2max - (A/0.001)*B_u3);
	} else if (DOflag == DOFLAG_OUTPUT) {
	} else if (DOflag == DOFLAG_FEATURE) {
		featurevector[0] = G;
		featurevector[1] = I;
		featurevector[2] = EPI;
		featurevector[3] = NOR;
		featurevector[4] = lactate;
		featurevector[5] = PVo2max;
		featurevector[6] = O2_deficit;
		featurevector[7] = lactate;
		featurevector[8] = met_stress;
		featurevector[9] = exercise_drive;
		featurevector[10] = NOR_neuronal;
		featurevector[11] = NOR_adrenal;
		featurevector[12] = Epi_adrenal;
		featurevector[13] = EPI;
		featurevector[14] = NOR;
		featurevector[15] = X;
		featurevector[16] = I;
		featurevector[17] = G;
		featurevector[18] = Gprod;
		featurevector[19] = Gup;
		featurevector[20] = Ie;
		featurevector[21] = Ggly;
		featurevector[22] = A;
	} else if (DOflag == DOFLAG_EVENT) {
		eventvector[0] = (gt(time, -INFINITY)) - 0.5;
	} else if (DOflag == DOFLAG_EVENTASSIGN) {
		if(eventstatus[0] == 1){
			statevector[1] = 0.0;
			statevector[3] = 0.0;
			statevector[2] = Lacb;
			statevector[8] = Norb;
			statevector[9] = Epib;
			statevector[10] = Ib;
			statevector[12] = Gb;
		}
	} else if (DOflag == DOFLAG_RESIDUAL) {
		RESvector[0] = timescale * (-0.8*PVo2max + 0.8*u3) - derivativevector[0];
		RESvector[1] = timescale * (max(vmax_o2 * max(PVo2max-threshold,0.0),0.0) - k_O2 * O2_deficit) - derivativevector[1];
		RESvector[2] = timescale * (vmax_lac * f_hill(max(O2_deficit,0.0), km_lac, n_lac) - elim_lactate*(lactate-Lacb)) - derivativevector[2];
		RESvector[3] = timescale * (f_2(max(PVo2max-threshold,0.0), k_stress) + lac_stress * max(lactate-Lacb,0.0) - elim_stress*met_stress) - derivativevector[3];
		RESvector[4] = timescale * ((k_push *  max(time - time_ex, 0.0) + k_push2) * PVo2max* max(sign(u3),0.0) - k_rec_drive * exercise_drive) - derivativevector[4];
		RESvector[5] = timescale * (Vmax_stres_n * max(met_stress,0.0) + Vmax_ex_n * f_mm(max(exercise_drive,0.0), km_ex_n) - spill * NOR_neuronal - elim_NOR_neuronal * NOR_neuronal) - derivativevector[5];
		RESvector[6] = timescale * (Vmax_stres_a_eff * f_mm(max(met_stress,0.0), km_stres_a) + Vmax_ex_a_eff * f_hill(max(exercise_drive,0.0), km_ex_a, n_ex_a) - release_nor_adrenal * NOR_adrenal - conv_nor_to_epi * NOR_adrenal) - derivativevector[6];
		RESvector[7] = timescale * (conv_nor_to_epi * NOR_adrenal - release_epi * Epi_adrenal) - derivativevector[7];
		RESvector[8] = timescale * (spill * NOR_neuronal + release_nor_adrenal * NOR_adrenal- elim_nor_plasma*(NOR-Norb)) - derivativevector[8];
		RESvector[9] = timescale * (release_epi * Epi_adrenal - elim_epi_plasma * (EPI-Epib)) - derivativevector[9];
		RESvector[10] = timescale * (k_I_stim*max(G-Gb,0.0)/(1.0+k_I_epi*max(EPI-Epib,0.0) + k_I_nor*max((NOR-Norb),0.0)) - elim_insu * (I- Ib) - Ie) - derivativevector[10];
		RESvector[11] = timescale * (ins_action *(I - Ib) - elim_X * X - Vmax_X_epi * f_mm(max((EPI-Epib),0.0), km_X_epi) * X) - derivativevector[11];
		RESvector[12] = timescale * ((BW/VolG)*(Gprod - Ggly) - (BW/VolG) * Gup - X*G - elim_glu *(G - Gb)) - derivativevector[12];
		RESvector[13] = timescale * (k_Gpro_stim * PVo2max - k_Gpro_decay * Gprod + epi_nor) - derivativevector[13];
		RESvector[14] = timescale * (K_Gup_stim  * PVo2max - k_Gup_decay * Gup) - derivativevector[14];
		RESvector[15] = timescale * (K_Ie_stim * PVo2max  - k_Ie_decay * Ie) - derivativevector[15];
		RESvector[16] = timescale * (gly_A*k*A_u3 - Ggly/T1*B_u3) - derivativevector[16];
		RESvector[17] = timescale * (A_u3*PVo2max - (A/0.001)*B_u3) - derivativevector[17];
	}
}

/* Function for initial condition definition */
static void M_glucose7_initialcondition(double *icvector, double *dericvector, double *parametervector, const std::vector<double>& inputs){
	[[maybe_unused]] double time{0.0};
	[[maybe_unused]] double vmax_thresh{parametervector[0]};
	[[maybe_unused]] double km_thresh{parametervector[1]};
	[[maybe_unused]] double vmax_o2{parametervector[2]};
	[[maybe_unused]] double k_O2{parametervector[3]};
	[[maybe_unused]] double vmax_lac{parametervector[4]};
	[[maybe_unused]] double km_lac{parametervector[5]};
	[[maybe_unused]] double n_lac{parametervector[6]};
	[[maybe_unused]] double elim_lactate{parametervector[7]};
	[[maybe_unused]] double k_stress{parametervector[8]};
	[[maybe_unused]] double elim_stress{parametervector[9]};
	[[maybe_unused]] double lac_stress{parametervector[10]};
	[[maybe_unused]] double k_push{parametervector[11]};
	[[maybe_unused]] double k_push2{parametervector[12]};
	[[maybe_unused]] double k_rec_drive{parametervector[13]};
	[[maybe_unused]] double Vmax_stres_n{parametervector[14]};
	[[maybe_unused]] double km_stres_n{parametervector[15]};
	[[maybe_unused]] double Vmax_ex_n{parametervector[16]};
	[[maybe_unused]] double km_ex_n{parametervector[17]};
	[[maybe_unused]] double spill{parametervector[18]};
	[[maybe_unused]] double elim_NOR_neuronal{parametervector[19]};
	[[maybe_unused]] double Vmax_stres_a{parametervector[20]};
	[[maybe_unused]] double km_stres_a{parametervector[21]};
	[[maybe_unused]] double Vmax_ex_a{parametervector[22]};
	[[maybe_unused]] double km_ex_a{parametervector[23]};
	[[maybe_unused]] double n_ex_a{parametervector[24]};
	[[maybe_unused]] double scale{parametervector[25]};
	[[maybe_unused]] double conv{parametervector[26]};
	[[maybe_unused]] double release_epi{parametervector[27]};
	[[maybe_unused]] double elim_nor_plasma{parametervector[28]};
	[[maybe_unused]] double elim_epi_plasma{parametervector[29]};
	[[maybe_unused]] double k_I_stim{parametervector[30]};
	[[maybe_unused]] double elim_insu{parametervector[31]};
	[[maybe_unused]] double k_I_epi{parametervector[32]};
	[[maybe_unused]] double k_I_nor{parametervector[33]};
	[[maybe_unused]] double Vmax_I_epi{parametervector[34]};
	[[maybe_unused]] double Vmax_I_nor{parametervector[35]};
	[[maybe_unused]] double km_I_epi{parametervector[36]};
	[[maybe_unused]] double km_I_nor{parametervector[37]};
	[[maybe_unused]] double n_I_epi{parametervector[38]};
	[[maybe_unused]] double n_I_nor{parametervector[39]};
	[[maybe_unused]] double elim_X{parametervector[40]};
	[[maybe_unused]] double ins_action{parametervector[41]};
	[[maybe_unused]] double Vmax_X_epi{parametervector[42]};
	[[maybe_unused]] double km_X_epi{parametervector[43]};
	[[maybe_unused]] double elim_glu{parametervector[44]};
	[[maybe_unused]] double k_Gpro_stim{parametervector[45]};
	[[maybe_unused]] double k_Gpro_decay{parametervector[46]};
	[[maybe_unused]] double Vmax_Gprod_epi{parametervector[47]};
	[[maybe_unused]] double Vmax_Gprod_nor{parametervector[48]};
	[[maybe_unused]] double km_Gprod_epi{parametervector[49]};
	[[maybe_unused]] double km_Gprod_nor{parametervector[50]};
	[[maybe_unused]] double n_Gprod_epi{parametervector[51]};
	[[maybe_unused]] double n_Gprod_nor{parametervector[52]};
	[[maybe_unused]] double K_Gup_stim{parametervector[53]};
	[[maybe_unused]] double k_Gup_decay{parametervector[54]};
	[[maybe_unused]] double K_Ie_stim{parametervector[55]};
	[[maybe_unused]] double k_Ie_decay{parametervector[56]};
	[[maybe_unused]] double k{parametervector[57]};
	[[maybe_unused]] double T1{parametervector[58]};

	[[maybe_unused]] const double u3{inputs[0]};
	[[maybe_unused]] const double Ib{inputs[1]};
	[[maybe_unused]] const double Gb{inputs[2]};
	[[maybe_unused]] const double BW{inputs[3]};
	[[maybe_unused]] const double Epib{inputs[4]};
	[[maybe_unused]] const double Norb{inputs[5]};
	[[maybe_unused]] const double Lacb{inputs[6]};
	[[maybe_unused]] const double time_ex{inputs[7]};
	[[maybe_unused]] const double Vo2max{inputs[8]};

	[[maybe_unused]] double release_nor_adrenal{conv*0.2};
	[[maybe_unused]] double conv_nor_to_epi{conv*0.8};
	[[maybe_unused]] double Vmax_stres_a_eff{Vmax_stres_a * (1.0+ scale * max(Vo2max-40.0,0.0))};
	[[maybe_unused]] double Vmax_ex_a_eff{Vmax_ex_a * (1.0+ scale * max(Vo2max-40.0,0.0))};
	[[maybe_unused]] double threshold{vmax_thresh * f_mm(Vo2max/90.0, km_thresh)};
	[[maybe_unused]] double VolG{11.7};

	[[maybe_unused]] const double PVo2max{0.0};
	[[maybe_unused]] const double O2_deficit{0.0};
	[[maybe_unused]] const double lactate{1.0};
	[[maybe_unused]] const double met_stress{0.0};
	[[maybe_unused]] const double exercise_drive{0.0};
	[[maybe_unused]] const double NOR_neuronal{0.0};
	[[maybe_unused]] const double NOR_adrenal{0.0};
	[[maybe_unused]] const double Epi_adrenal{0.0};
	[[maybe_unused]] const double NOR{1.3};
	[[maybe_unused]] const double EPI{0.3};
	[[maybe_unused]] const double I{12.0};
	[[maybe_unused]] const double X{0.0};
	[[maybe_unused]] const double G{4.9750};
	[[maybe_unused]] const double Gprod{0.0};
	[[maybe_unused]] const double Gup{0.0};
	[[maybe_unused]] const double Ie{0.0};
	[[maybe_unused]] const double Ggly{0.0};
	[[maybe_unused]] const double A{0.0};

	[[maybe_unused]] double ATH{(-1.1521*pow(PVo2max,2.0))+(87.471*PVo2max)};
	[[maybe_unused]] double gly_A{(sign(A-ATH+1.0)+1.0)/2.0};
	[[maybe_unused]] double A_u3{1.0*sign(PVo2max)};
	[[maybe_unused]] double B_u3{(sign(1.0-PVo2max)+1.0)/2.0};
	[[maybe_unused]] double I_epi_nor{Vmax_I_epi * f_hill(max((EPI-Epib),0.0), km_I_epi, n_I_epi) + Vmax_I_nor * f_hill(max((NOR-Norb),0.0), km_I_nor, n_I_nor)};
	[[maybe_unused]] double epi_nor{Vmax_Gprod_epi * f_hill(max((EPI-Epib),0.0), km_Gprod_epi, n_Gprod_epi) + Vmax_Gprod_nor * f_hill(max((NOR-Norb),0.0), km_Gprod_nor, n_Gprod_nor)};

	icvector[0] = PVo2max;
	icvector[1] = O2_deficit;
	icvector[2] = lactate;
	icvector[3] = met_stress;
	icvector[4] = exercise_drive;
	icvector[5] = NOR_neuronal;
	icvector[6] = NOR_adrenal;
	icvector[7] = Epi_adrenal;
	icvector[8] = NOR;
	icvector[9] = EPI;
	icvector[10] = I;
	icvector[11] = X;
	icvector[12] = G;
	icvector[13] = Gprod;
	icvector[14] = Gup;
	icvector[15] = Ie;
	icvector[16] = Ggly;
	icvector[17] = A;

	dericvector[0] = 0;
	dericvector[1] = 0;
	dericvector[2] = 0;
	dericvector[3] = 0;
	dericvector[4] = 0;
	dericvector[5] = 0;
	dericvector[6] = 0;
	dericvector[7] = 0;
	dericvector[8] = 0;
	dericvector[9] = 0;
	dericvector[10] = 0;
	dericvector[11] = 0;
	dericvector[12] = 0;
	dericvector[13] = 0;
	dericvector[14] = 0;
	dericvector[15] = 0;
	dericvector[16] = 0;
	dericvector[17] = 0;
}

/* Functions for Python C API Extension */
static PyTypeObject M_glucose7_Type = {
	.ob_base = PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name =  "sund.Models.M_glucose7.M_glucose7",
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	.tp_doc = "This model describes the glucose and insulin dynamics during excercise\nPlamsa insulin (I),\nRemote Insulin (X),\nPlasma glucose (G),\nGlucose production (Gprod),\nGlucose uptake (Gup),\nInsulin elimination due to exercise (Ie),\nPercentage of maximal oxygen consumption (PVo2max),\nDecline in glucose production by glycogenolyse (Ggly),\nIntegrated excercise intensity (A)\n\nEpi and nor influence on Gprod made hill instead of mm\nX not maximal 0\n\nAuthor(s): unknown\n",
	.tp_new = M_glucose7_new,
};

static ModelStructure M_glucose7 = {
	.function = M_glucose7_modelfunction,
	.initialcondition = M_glucose7_initialcondition,
	.numberof = M_glucose7_numberof,
	.statenames = M_glucose7_statenames,
	.featurenames = M_glucose7_featurenames,
	.featureunits = M_glucose7_featureunits,
	.outputnames = M_glucose7_outputnames,
	.inputnames = M_glucose7_inputnames,
	.parameternames = M_glucose7_parameternames,
	.eventnames = M_glucose7_eventnames,
	.differentialstates = M_glucose7_differentialstates,
	.inputdependency = M_glucose7_inputdependency,
	.defaultparameters = M_glucose7_defaultparameters,
	.timeunit = M_glucose7_timeunit,
	.has_algebraic_eq = 0,
	.mandatoryinputs = M_glucose7_mandatoryinputs,
	.defaultInputs = M_glucose7_defaultInputs
};

static PyObject *
M_glucose7_new(PyTypeObject *type, PyObject *args, PyObject *kwds){
	ModelObject *self;
	self = (ModelObject *) type->tp_alloc(type, 0);
	if (self) {
		self->model = &M_glucose7;
		if(Model_alloc(self) < 0){
			Py_DECREF(self);
			return NULL;
		}
	}
	return (PyObject *) self;
}

static PyModuleDef M_glucose7Module = {
	.m_base = PyModuleDef_HEAD_INIT,
	.m_name = "sund.Models.M_glucose7",
	.m_doc = "M_glucose7 Module",
	.m_size = -1
};

PyMODINIT_FUNC
PyInit_M_glucose7(void){
	PyObject *m;

	m = PyModule_Create(&M_glucose7Module);
	if (m == NULL)
		return NULL;

	import_ModelCoreAPI();
	M_glucose7_Type.tp_base = Model_Base_Type;
	if (PyType_Ready(&M_glucose7_Type) < 0){
		Py_DECREF(m);
		return NULL;
	}

	Py_INCREF(&M_glucose7_Type);
	if (PyModule_AddObject(m, "M_glucose7", (PyObject *) &M_glucose7_Type) < 0){
		Py_DECREF(&M_glucose7_Type);
		Py_DECREF(m);
		return NULL;
	}

	return m;
}