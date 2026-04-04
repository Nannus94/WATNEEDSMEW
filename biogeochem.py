import numpy as np
import pyEW
import warnings
from scipy import optimize
from scipy.optimize import fsolve
#minimize, least_squares, newton_krylov, broyden1, root, broyden2
def biogeochem_balance(n, s, L, T, I, v, k_v, RAI, root_d, Zr, r_het, r_aut, D, temp_soil, pH_in, conc_in, f_CEC_in, K_CEC, CEC_tot, Si_in, CaCO3_in, MgCO3_in, M_rock_in, t_app, mineral, rock_f_in, d_in, psd_perc_in, SSA_in, diss_f, dt, conv_Al, conv_mol, keyword_add, I_background=None, debug_label=None, use_mean_s_init=False, monitor_progress=False, An_in=None, reset_carbonates_at_step=None):
 
    # DEBUG: Print input concentrations and pH
    # Debug output disabled for clean production runs
 print(f"BIOGEOCHEM DEBUG: pH={pH_in:.2f}, conc_in=[Ca:{conc_in[0]:.1f}, Mg:{conc_in[1]:.1f}, K:{conc_in[2]:.1f}, Na:{conc_in[3]:.1f}, Al:{conc_in[4]:.1f}]")
    
    # Preallocating the variables
    pH = np.zeros(len(s))
    H = np.zeros(len(s))
    f_H = np.zeros(len(s))
    
    Ca_tot = np.zeros(len(s))
    Ca = np.zeros(len(s))
    f_Ca = np.zeros(len(s))
    UP_Ca = np.zeros(len(s))
    
    Omega_CaCO3 = np.zeros(len(s))
    CaCO3 = np.zeros(len(s))
    W_CaCO3 = np.zeros(len(s))
    
    Mg_tot = np.zeros(len(s))
    Mg = np.zeros(len(s))
    f_Mg = np.zeros(len(s))
    UP_Mg = np.zeros(len(s))
    
    Omega_MgCO3 = np.zeros(len(s))
    MgCO3 = np.zeros(len(s))
    W_MgCO3 = np.zeros(len(s))
    
    K_tot = np.zeros(len(s))
    K = np.zeros(len(s))
    f_K = np.zeros(len(s))
    UP_K = np.zeros(len(s))
    
    Na_tot = np.zeros(len(s))
    Na = np.zeros(len(s))
    f_Na = np.zeros(len(s))
    
    Si = np.zeros(len(s))
    Si_tot = np.zeros(len(s))
    UP_Si = np.zeros(len(s))
    
    root_ex = np.zeros([len(s)])
    
    An = np.zeros([len(s)])
    An_tot = np.zeros([len(s)])
    R_alk = np.zeros(len(s))
    Alk_tot = np.zeros(len(s))
    Alk = np.zeros(len(s))
    
    CO2_air = np.zeros(len(s))
    IC_tot = np.zeros(len(s))
    CO2_w = np.zeros(len(s))
    HCO3 = np.zeros(len(s))
    CO3 = np.zeros(len(s))
    DIC = np.zeros(len(s))
    Fs = np.zeros(len(s))
    ADV = np.zeros(len(s))
       
    H_rain = np.zeros(len(s))
    DIC_rain = np.zeros(len(s))
    
    Al = np.zeros(len(s))
    f_Al = np.zeros(len(s))
    AlOH = np.zeros(len(s))
    AlOH2 = np.zeros(len(s))
    AlOH3 = np.zeros(len(s)) 
    AlOH4 = np.zeros(len(s))
    Al_w = np.zeros(len(s))
    Al_tot = np.zeros(len(s))
    
    M_rock = np.zeros(len(s), dtype=np.float32)
    SA = np.zeros(len(s), dtype=np.float32)
    EW = np.zeros([1, len(s)], dtype=np.float32)
    min_st = np.zeros([1, 6])
    
    d = np.zeros([1, len(s)], dtype=np.float32)
    delta_d = np.zeros([1, len(s)], dtype=np.float32)
    lamb = np.zeros([1, len(s)], dtype=np.float32)
    SSA = np.zeros([1, len(s)], dtype=np.float32)
    psd = np.zeros([1, len(s)], dtype=np.float32)
     
    if M_rock_in > 0:
        number_min = len(mineral)
        rho_min = np.zeros(number_min)
        MM_min = np.zeros(number_min)
        E_H = np.zeros(number_min) 
        E_w = np.zeros(number_min)
        E_OH = np.zeros(number_min)
        n_H = np.zeros(number_min)
        n_OH = np.zeros(number_min)
        K_sp = np.zeros(number_min)
        min_st = np.zeros([number_min, 6])

        k_diss_H_t = np.zeros([number_min, len(s)])
        k_diss_w_t = np.zeros([number_min, len(s)])
        k_diss_OH_t = np.zeros([number_min, len(s)])
        Omega =  np.zeros([number_min, len(s)])
        M_min = np.zeros([number_min, len(s)], dtype=np.float32)
        rock_f = np.zeros([number_min, len(s)], dtype=np.float32)
        Wr = np.zeros([number_min, len(s)], dtype=np.float32)
        EW = np.zeros([number_min, len(s)], dtype=np.float32)
        
        n_d_cl = len(d_in)
        if n_d_cl > 1:
            d = np.zeros([n_d_cl, len(s)], dtype=np.float32)
            delta_d = np.zeros([n_d_cl, len(s)], dtype=np.float32)
            lamb = np.zeros([n_d_cl, len(s)], dtype=np.float32)
            SSA = np.zeros([n_d_cl, len(s)], dtype=np.float32)
            psd = np.zeros([n_d_cl, len(s)], dtype=np.float32)
        # Track cumulative applied rock mass for clearer logs
        applied_cum = 0.0
        
    errors = np.zeros([16, len(s)]) 
    
#------------------------------------------------------------------------------
    # Constants
    CO2_atm = pyEW.CO2_atm(conv_mol) # [mol_CO2/l_air] Atmospheric CO2 concentration
    T_K = temp_soil + 273.15
    
    # soil CO2 diffusivity - USE THE PASSED PARAMETER (calculated by carbon respiration)
    # D_0 = pyEW.D_0() #free-air diffusion [m2/d]
    # D = D_0*(1-s)**(10/3)*n**(4/3) #Mill-Quirk (1961)  # COMMENTED OUT - was overwriting passed parameter
    
    # DEBUG: Print the D[0] value that biogeochem receives and uses
    
    #solute diffusivity in soil water
    Dw_0 = pyEW.Dw_0()
    Dw = Dw_0*(n*s)**2 # Archie 1942, Grathwohl 1998 (book)
    
    # [g/mol-conv]: Molar masses    
    [MM_Mg, MM_Ca, MM_Na, MM_K, MM_Si, MM_C, MM_Anions, MM_Al]=pyEW.MM(conv_mol) 
    
    # Aluminium speciation
    [K1, K2, K3, K4] = pyEW.K_Al(conv_mol) 
    
    # carbonate spec  
    [k1, k2, k_w, k_H] = pyEW.K_C(T_K,conv_mol)
 
    #CEC Gaines-Thomas constants
    [K_Ca_Mg, K_Ca_K, K_Ca_Na, K_Ca_Al, K_Ca_H]  = K_CEC
    
    #nutrient uptake by plants
    [v_f_Ca, v_f_Mg, v_f_K, v_f_Si] = pyEW.plant_nutr_f()
    dry_perc = 0.1 #percent of dry mass
    xi = dry_perc*np.array([v_f_Ca/MM_Ca, v_f_Mg/MM_Mg, v_f_K/MM_K, v_f_Si/MM_Si]) # [mol-conv/g_biomass]
    
    #carb weathering constants
    [K_CaCO3,K_MgCO3,r_CaCO3,r_MgCO3,tau_CaCO3,tau_MgCO3] = pyEW.carb_weath_const(conv_mol)
    
    #mineral constants
    if M_rock_in > 0: 
        for j in range(0,number_min):
            MM_min[j], k_diss_H_t[j,:], k_diss_w_t[j,:], k_diss_OH_t[j,:], n_H[j], n_OH[j], min_st[j,:], K_sp[j] = pyEW.min_const(mineral[j], T_K, conv_mol)  
    
    #rock density
    rho_rock = 3*1e6 # [g/m3]
    
    #rock surface fractality (Beerling 2020)
    b = 0.35 #[-]
    a = (1/(2*1e-10))**b #[1/m^b]
    
#------------------------------------------------------------------------------
    # RAINWATER
    
    Alk_rain = 0 #alk
    CO2_w_rain = k_H*CO2_atm # [mol/l] Henry's law
    
    for i in range(0, len(s)):
        def equations(p):
            H_rain[i] = p
            return(Alk_rain-(k1[i]*CO2_w_rain[i]/H_rain[i]+2*k1[i]*k2[i]*CO2_w_rain[i]/(H_rain[i]**2)-H_rain[i]+k_w[i]/H_rain[i]))
        
        H_rain[i] = fsolve(equations, 10**-6*conv_mol) # [mol/l]
        DIC_rain[i]=CO2_w_rain[i]+k1[i]*CO2_w_rain[i]/H_rain[i]+k2[i]*k1[i]*CO2_w_rain[i]/(H_rain[i]**2)
    
#------------------------------------------------------------------------------            
    # INITIAL CONDITIONS
    
    # Implicit i=0 initialization: solve full system with pH free, pH_in as initial guess.
    # pCO2 geometry
    if Zr <= 0.3:
        Z_CO2 = Zr/2
    else:
        Z_CO2 = 0.15
    # Initial CO2: use timestep-0 values (must match caller's initialization)
    # D gets a first-week average + floor to prevent singularity when s[0]~1
    steps_per_day = int(1.0/dt) if dt > 0 else 24
    n0 = min(len(D), steps_per_day * 7)
    D_eff0 = max(float(np.mean(D[:n0])), 0.01)
    CO2_air_init = (float(r_het[0]) + float(r_aut[0]))/(D_eff0*1000/(Z_CO2)) + CO2_atm
    CO2_w_init = k_H[0]*CO2_air_init
    # Initial guesses from mapped pH
    pH_guess = float(pH_in)
    H_guess = 10**(-pH_guess)*conv_mol
    HCO3_guess = k1[0]*CO2_w_init/H_guess
    CO3_guess  = k2[0]*k1[0]*CO2_w_init/(H_guess**2)
    Alk_guess  = HCO3_guess + 2*CO3_guess - H_guess + k_w[0]/H_guess
    # cation guesses and adsorbed fractions
    [Ca[0], Mg[0], K[0], Na[0], Al_w[0]] = conc_in
    [f_Ca[0], f_Mg[0], f_K[0], f_Na[0], f_Al[0], f_H[0]] = f_CEC_in
    R_alk[0] = (f_Mg[0]+f_Ca[0]+f_Na[0]+f_K[0])*CEC_tot
    # OPTION: Use mean soil moisture for initialization (more stable for calibration)
    if use_mean_s_init:
        s0 = float(np.mean(s))  # Use average moisture → reduces sensitivity to initial fluctuations
    else:
        s0 = float(s[0])  # Use actual initial moisture (default)
    # Use provided An concentration or default to 1.0
    target_An0 = float(An_in) if An_in is not None else 1.0
    Ca_tot[0] = Ca[0]*n*s0*Zr*1000+f_Ca[0]/2*CEC_tot
    Mg_tot[0] = Mg[0]*n*s0*Zr*1000+f_Mg[0]/2*CEC_tot
    K_tot[0]  = K[0]*n*s0*Zr*1000+f_K[0]*CEC_tot
    Na_tot[0] = Na[0]*n*s0*Zr*1000+f_Na[0]*CEC_tot
    An_tot[0] = target_An0*n*s0*Zr*1000
    Al_tot[0] = Al_w[0]*n*Zr*s0*1000+(f_Al[0]/3)*CEC_tot*conv_Al
    Si[0] = Si_in
    Si_tot[0] = Si[0]*n*Zr*s0*1000
    DIC_init = HCO3_guess + CO3_guess + CO2_w_init
    IC_tot[0] = (DIC_init*s0 + CO2_air_init*(1 - s0))*(n*Zr*1000)
    # Build Alk_tot from totals
    Alk_tot[0] = 2*Mg_tot[0]+2*Ca_tot[0]+Na_tot[0]+K_tot[0]-An_tot[0]
    # Solve implicit system at i=0
    Alk0 = Alk_guess; CO2_w0 = CO2_w_init; H0 = H_guess; R_alk0 = R_alk[0]
    Al0 = (H0**4/(H0**4+H0**3*K1+H0**2*K1*K2+H0*K1*K2*K3+K1*K2*K3*K4))*Al_w[0]
    Mg0 = Mg[0]; Ca0 = Ca[0]; Na0 = Na[0]; K0 = K[0]
    x0 = np.array([Alk0, CO2_w0, H0, R_alk0, Al_w[0], Al0, Mg0, Ca0, Na0, K0, f_Al[0], f_Mg[0], f_Na[0], f_K[0], f_H[0], f_Ca[0]])
    def equations0(p):
        Alk[0], CO2_w[0], H[0], R_alk[0], Al_w[0], Al[0], Mg[0], Ca[0], Na[0], K[0], f_Al[0], f_Mg[0], f_Na[0], f_K[0], f_H[0], f_Ca[0] = p
        return (
            (Alk_tot[0]-R_alk[0]) - Alk[0]*(n*Zr*s0*1000),
            IC_tot[0]-(CO2_w[0]*(1+k1[0]/H[0]+k2[0]*k1[0]/(H[0]**2))*s0+(CO2_w[0]/k_H[0])*(1-s0))*(n*Zr*1000),
            (k1[0]*CO2_w[0]/H[0]+2*k1[0]*k2[0]*CO2_w[0]/(H[0]**2)-H[0]+k_w[0]/H[0]) - Alk[0],
            R_alk[0] - (f_Mg[0]+f_Ca[0]+f_Na[0]+f_K[0])*CEC_tot,
            Al_w[0]*n*Zr*s0*1000+(f_Al[0]/3)*CEC_tot*conv_Al - Al_tot[0],
            Al[0]-(H[0]**4/(H[0]**4+H[0]**3*K1+H[0]**2*K1*K2+H[0]*K1*K2*K3+K1*K2*K3*K4))*Al_w[0],
            Mg[0]*n*Zr*s0*1000+f_Mg[0]/2*CEC_tot - Mg_tot[0],
            Ca[0]*n*Zr*s0*1000+f_Ca[0]/2*CEC_tot - Ca_tot[0],
            Na[0]*n*Zr*s0*1000+f_Na[0]*CEC_tot - Na_tot[0],
            K[0]*n*Zr*s0*1000+f_K[0]*CEC_tot - K_tot[0],
            f_Al[0] - (Al[0]/conv_Al)*(f_Ca[0]**3/(K_Ca_Al*Ca[0]**3))**(1/2),
            f_Mg[0] - Mg[0]*(f_Ca[0]/(K_Ca_Mg*Ca[0])),
            f_Na[0] - Na[0]*(f_Ca[0]/(K_Ca_Na*Ca[0]))**(1/2),
            f_K[0]  - K[0] *(f_Ca[0]/(K_Ca_K*Ca[0]))**(1/2),
            f_H[0]  - H[0] *(f_Ca[0]/(K_Ca_H*Ca[0]))**(1/2),
            1-(f_Ca[0]+f_Al[0]+f_Mg[0]+f_Na[0]+f_K[0]+f_H[0])
        )
    sol0 = fsolve(equations0, x0, xtol=1e-8)
    Alk[0], CO2_w[0], H[0], R_alk[0], Al_w[0], Al[0], Mg[0], Ca[0], Na[0], K[0], f_Al[0], f_Mg[0], f_Na[0], f_K[0], f_H[0], f_Ca[0] = sol0
    # Derived species and flux
    pH[0] = -np.log10(H[0]/conv_mol)
    CO2_air[0] = CO2_w[0]/k_H[0]
    HCO3[0] = k1[0]*CO2_w[0]/H[0]
    CO3[0]  = k2[0]*k1[0]*CO2_w[0]/(H[0]**2)
    DIC[0]  = CO2_w[0]+HCO3[0]+CO3[0]
    Fs[0]   = D[0]/(Z_CO2)*(CO2_air[0]-CO2_atm)*1000
    # Set An[0] from target_An0 (needed for An_tot[0] recalculation later)
    An[0] = target_An0
    # DEBUG: Print alkalinity calculation details  
    biogeochem_cation_charge = 2*Mg[0]+2*Ca[0]+Na[0]+K[0]
        
    # aluminium speciation
    Al[0]=(H[0]**4/(H[0]**4+H[0]**3*K1+H[0]**2*K1*K2+H[0]*K1*K2*K3+K1*K2*K3*K4))*Al_w[0] #mol/l
    AlOH[0]=(H[0]**3*K1/(H[0]**4+H[0]**3*K1+H[0]**2*K1*K2+H[0]*K1*K2*K3+K1*K2*K3*K4))*Al_w[0]
    AlOH2[0]=(H[0]**2*K1*K2/(H[0]**4+H[0]**3*K1+H[0]**2*K1*K2+H[0]*K1*K2*K3+K1*K2*K3*K4))*Al_w[0]
    AlOH3[0]=(H[0]*K1*K2*K3/(H[0]**4+H[0]**3*K1+H[0]**2*K1*K2+H[0]*K1*K2*K3+K1*K2*K3*K4))*Al_w[0]
    AlOH4[0]=Al_w[0]-(Al[0]+AlOH[0]+AlOH2[0]+AlOH3[0])
    
    # Silicon
    Si[0] = Si_in
    
    # Background inputs (rain, litterfall, background weathering..)
    # Constant in time, scaled by mean(L) only (neutral vs ET), scalar style
    # -------------------
    # Background inputs (rain, litterfall, background weathering..)
    # Accepts I_background only as a dict with keys:
    #   'I_Ca','I_Mg','I_K','I_Na','I_Si','I_An'  (units: mol d-1 per soil column)
    # Priority: explicit I_background (if provided) > keyword_add behavior.
    # If keyword_add == 0, I_background is required (avoid silent-zero).
    # -------------------

    # Defensive: ensure I_background is either None or a dict
    if I_background is not None and not isinstance(I_background, dict):
        raise TypeError("I_background must be a dict with keys 'I_Ca','I_Mg','I_K','I_Na','I_Si','I_An' (or None).")

    # If the caller requested keyword_add == 0, require explicit I_background to avoid silent zeros
    if keyword_add == 0 and I_background is None:
        raise ValueError(
            "I_background must be provided when keyword_add == 0. "
            "Provide a dict: {'I_Ca':..., 'I_Mg':..., 'I_K':..., 'I_Na':..., 'I_Si':..., 'I_An':...} "
            "(units: mol d-1 per soil column)."
        )

    # If explicit background provided, use it (clamp negatives -> 0)
    if I_background is not None:
        try:
            # Extract and clamp to non-negative
            I_Ca = max(0.0, float(I_background.get('I_Ca', 0.0)))
            I_Mg = max(0.0, float(I_background.get('I_Mg', 0.0)))
            I_K  = max(0.0, float(I_background.get('I_K', 0.0)))
            I_Na = max(0.0, float(I_background.get('I_Na', 0.0)))
            I_Si = max(0.0, float(I_background.get('I_Si', 0.0)))
            I_An = max(0.0, float(I_background.get('I_An', 0.0)))
            I_Al = max(0.0, float(I_background.get('I_Al', 0.0)))
        except Exception:
            # Fallback to zeros and warn when debug_label present
            I_Ca = I_Mg = I_K = I_Na = I_Si = I_An = I_Al = 0.0
            if debug_label is not None:
                print(f"[biogeochem_balance] Warning: malformed I_background for {debug_label}; using zeros.")
    else:
        # No explicit I_background given -> fall back to legacy keyword_add logic
        if keyword_add == 1:
            factor = np.mean(L + T) * 1000.0  # original scaling [m/d] -> mol d-1 per column
            I_An = factor * An[0]  # [mol_c d-1]
            I_Ca = factor * Ca[0]  # [mol d-1]
            I_Mg = factor * Mg[0]
            I_Na = factor * Na[0]
            I_K  = factor * K[0]
            I_Si = factor * Si[0]
            I_Al = factor * Al[0]
        else:
            # This branch should not occur because of the earlier enforcement,
            # but keep a safe fallback to zeros.
            I_An = 0.0
            I_Ca = 0.0
            I_Mg = 0.0
            I_Na = 0.0
            I_K = 0.0
            I_Si = 0.0
            I_Al = 0.0

    # (Now I_Ca, I_Mg, I_K, I_Na, I_Si, I_An, I_Al are defined and used later in mass balances.)

        print(f"[BG weathering] I_An: {I_An}, I_Ca: {I_Ca}, I_Mg: {I_Mg}, I_Na: {I_Na}, I_K: {I_K}, I_Si: {I_Si}")

    #CEC adsorbed species
    [f_Ca[0], f_Mg[0], f_K[0], f_Na[0], f_Al[0], f_H[0]] = f_CEC_in
    
    #reserve of alkalinity
    R_alk[0] = (f_Mg[0]+f_Ca[0]+f_Na[0]+f_K[0])*CEC_tot # [mol_c]
    
    #total amounts (solution and adsorbed)
    Ca_tot[0] = Ca[0]*n*s0*Zr*1000+f_Ca[0]/2*CEC_tot # [mol] 
    Mg_tot[0] = Mg[0]*n*s0*Zr*1000+f_Mg[0]/2*CEC_tot # [mol] 
    K_tot[0] = K[0]*n*s0*Zr*1000+f_K[0]*CEC_tot # [mol]
    Na_tot[0] = Na[0]*n*s0*Zr*1000+f_Na[0]*CEC_tot # [mol]
    Alk_tot[0] = 2*Mg_tot[0]+2*Ca_tot[0]+Na_tot[0]+K_tot[0]-An[0]*(n*s0*Zr*1000) # [mol_c]
    An_tot[0] = An[0]*n*s0*Zr*1000 #[mol_c]
    Al_tot[0] = Al_w[0]*n*Zr*s0*1000+(f_Al[0]/3)*CEC_tot*conv_Al # [mol]
    Si_tot[0] = Si[0]*n*Zr*s0*1000
    
    #Carbonate minerals (considered as an additional pool)
    CaCO3[0] = CaCO3_in # [mol-conv]
    MgCO3[0] = MgCO3_in
    
    #Carbonate weathering
    Omega_CaCO3[0] = Ca[0]*CO3[0]/K_CaCO3 # [-]
    Omega_MgCO3[0] = Mg[0]*CO3[0]/K_MgCO3
    [W_CaCO3[0], W_MgCO3[0]] = pyEW.carb_W(CaCO3[0], MgCO3[0], Omega_CaCO3[0], Omega_MgCO3[0], s[0], Zr, r_CaCO3,r_MgCO3,tau_CaCO3,tau_MgCO3) # [mol-conv/ m2 d]
        
    #Silicate weathering
    if M_rock_in > 0:
        
        #application timestep
        # Se t_app è già un array o una lista, convertiamo ogni tempo in indice
        if np.isscalar(t_app):
            tt_app = [int(t_app/dt)]
        else:
            tt_app = [int(t/dt) for t in t_app]

       
        # Loop su ciascun tempo di applicazione
        for t_idx in tt_app:
            if t_idx < len(s):  # Assicuriamoci che l'indice sia valido
                # Aggiungi nuova roccia in quel timestep
                M_rock[t_idx] = M_rock[t_idx] + M_rock_in
                # Imposta la composizione: le frazioni minerali
                rock_f[:, t_idx] = rock_f_in
                # Aggiorna la massa dei minerali aggiungendo la frazione corrispondente
                M_min[:, t_idx] = M_min[:, t_idx] + rock_f_in * M_rock_in
                # Calcola la porzione inerte (se le frazioni non sommano a 1)
                M_iner = M_rock[t_idx] * (1 - np.sum(rock_f_in))
                
                # Imposta le classi di diametro
                d[:, t_idx] = d_in
                delta_d[:, t_idx] = np.insert(np.diff(d[:, t_idx]), 0, d[0, t_idx])
                
                # Calcola la distribuzione delle dimensioni delle particelle
                psd[:, t_idx] = psd_perc_in * M_rock[t_idx] / delta_d[:, t_idx]
                
                # Se è disponibile un SSA misurato, raffina la costante frattale
                if SSA_in > 0:
                    a = (SSA_in * rho_rock * M_rock[t_idx] / 6) / np.sum(d[:, t_idx]**(b-1) * psd[:, t_idx] * delta_d[:, t_idx])
                
                # Calcola l'area superficiale
                lamb[:, t_idx] = a * d[:, t_idx]**b
                # SAFETY CHECK: Prevent division by zero in initial SSA calculation
                SSA[:, t_idx] = np.where(d[:, t_idx] > 1e-12, 
                                       6 / (d[:, t_idx] * rho_rock) * lamb[:, t_idx], 0)
                SA[t_idx] = np.sum(SSA[:, t_idx] * psd[:, t_idx] * delta_d[:, t_idx])
                
                # Se questo è il primo tempo di applicazione (tipicamente t=0), inizializza il weathering
                if t_idx == tt_app[0]:
                    for j in range(0, number_min):
                        Omega[j, 0] = pyEW.Omega_sil(mineral[j], Ca[0], Mg[0], K[0], Na[0], Si[0], H[0], K_sp[j], conv_mol)
                        Wr[j, 0] = s[0] * diss_f * (k_diss_H_t[j, 0] * (H[0] / conv_mol)**n_H[j] +
                                                    k_diss_w_t[j, 0] +
                                                    k_diss_OH_t[j, 0] * (H[0] / conv_mol)**n_OH[j]) * (1 - Omega[j, 0])
                        EW[j, 0] = Wr[j, 0] * SA[0] * rock_f[j, 0]
                if t_idx == tt_app[0]: # If this is the first application time being processed
                    if j == 0: # Print only once for this event (e.g., for the first mineral)
                         print(f"[ROCK] init EW set at t_idx={t_idx}")


#------------------------------------------------------------------------------
    #SYSTEM RESOLUTION
        
    printcounter = 0
    numb_print = 0
    n_solver_failures = 0
    max_solver_failures = int(0.05 * len(s))  # allow up to 5% failed timesteps

    for i in range(1, len(s)): 
        
            #CO2 advection due to moisture variation    
            if s[i]<s[i-1]:
                ADV[i] = n*Zr*1000*(s[i]-s[i-1])*CO2_atm # [mol] 
            elif s[i]>s[i-1]:
                ADV[i] = n*Zr*1000*(s[i]-s[i-1])*CO2_air[i-1]
            
            #active uptake [Ca, Mg, K, Si] 
            UP_act = pyEW.up_act(v[i], (v[i]-v[i-1]), xi, dt, T[i-1], Ca[i-1], Mg[i-1], K[i-1], Si[i-1], Dw[i-1], Zr, k_v, RAI, root_d)
            UP_Ca[i-1], UP_Mg[i-1], UP_K[i-1], UP_Si[i-1] = UP_act # [mol-conv/d] 
                                  
            #explicit mass balances # [mol]
            Ca_tot[i] = Ca_tot[i-1]+(I_Ca+np.sum(min_st[:,0]*EW[:,i-1])+W_CaCO3[i-1]-(L[i-1]+T[i-1])*1000*Ca[i-1]-UP_Ca[i-1])*dt 
            Mg_tot[i] = Mg_tot[i-1]+(I_Mg+np.sum(min_st[:,1]*EW[:,i-1])+W_MgCO3[i-1]-(L[i-1]+T[i-1])*1000*Mg[i-1]-UP_Mg[i-1])*dt
            K_tot[i] = K_tot[i-1]+(I_K+np.sum(min_st[:,2]*EW[:,i-1])-(L[i-1]+T[i-1])*1000*K[i-1]-UP_K[i-1])*dt
            Na_tot[i] = Na_tot[i-1]+(I_Na+np.sum(min_st[:,3]*EW[:,i-1])-(L[i-1]+T[i-1])*1000*Na[i-1])*dt
            Al_tot[i] = Al_tot[i-1]+(I_Al+np.sum(min_st[:,4]*EW[:,i-1])*conv_Al-L[i-1]*1000*(Al[i-1]+AlOH4[i-1]))*dt
            Si_tot[i] = Si_tot[i-1]+(I_Si+np.sum(min_st[:,5]*EW[:,i-1])-(L[i-1]+T[i-1])*1000*Si[i-1]-UP_Si[i-1])*dt
            # Conservative anions — constant concentration, only varies with moisture
            An[i] = An[0]
            An_tot[i] = An[i] * n * s[i] * Zr * 1000  # [mol_c]
            Alk_tot[i] = 2*Mg_tot[i]+2*Ca_tot[i]+Na_tot[i]+K_tot[i]-An_tot[i] # [mol_c]
            IC_tot[i] = IC_tot[i-1]+I[i]*1000*DIC_rain[i]-ADV[i]+(W_CaCO3[i-1]+W_MgCO3[i-1]+r_het[i-1]+r_aut[i-1]-Fs[i-1]-L[i-1]*1000*DIC[i-1])*dt 
                       
            #implicit system
            def equations(p):
                Alk[i], CO2_w[i], H[i], R_alk[i], Al_w[i], Al[i], Mg[i], Ca[i], Na[i], K[i], f_Al[i], f_Mg[i], f_Na[i], f_K[i], f_H[i], f_Ca[i] = p
            
                return((Alk_tot[i]-R_alk[i])-Alk[i]*(n*Zr*s[i]*1000),\
                       IC_tot[i]-(CO2_w[i]*(1+k1[i]/H[i]+k2[i]*k1[i]/(H[i]**2))*s[i]+(CO2_w[i]/k_H[i])*(1-s[i]))*(n*Zr*1000),\
                       (k1[i]*CO2_w[i]/H[i]+2*k1[i]*k2[i]*CO2_w[i]/(H[i]**2)-H[i]+k_w[i]/H[i])-Alk[i],\
                       R_alk[i]-(f_Mg[i]+f_Ca[i]+f_Na[i]+f_K[i])*CEC_tot,\
                       Al_w[i]*n*Zr*s[i]*1000+(f_Al[i]/3)*CEC_tot*conv_Al-Al_tot[i],\
                       Al[i]-(H[i]**4/(H[i]**4+H[i]**3*K1+H[i]**2*K1*K2+H[i]*K1*K2*K3+K1*K2*K3*K4))*Al_w[i],\
                       Mg[i]*n*Zr*s[i]*1000+f_Mg[i]/2*CEC_tot-Mg_tot[i],\
                       Ca[i]*n*Zr*s[i]*1000+f_Ca[i]/2*CEC_tot-Ca_tot[i],\
                       Na[i]*n*Zr*s[i]*1000+f_Na[i]*CEC_tot-Na_tot[i],\
                       K[i]*n*Zr*s[i]*1000+f_K[i]*CEC_tot-K_tot[i],\
                       f_Al[i] - (Al[i]/conv_Al)*(f_Ca[i]**3/(K_Ca_Al*Ca[i]**3))**(1/2),\
                       f_Mg[i] - Mg[i]*(f_Ca[i]/(K_Ca_Mg*Ca[i])),\
                       f_Na[i] - Na[i]*(f_Ca[i]/(K_Ca_Na*Ca[i]))**(1/2),\
                       f_K[i] - K[i]*(f_Ca[i]/(K_Ca_K*Ca[i]))**(1/2),\
                       f_H[i] - H[i]*(f_Ca[i]/(K_Ca_H*Ca[i]))**(1/2),\
                       1-(f_Ca[i]+f_Al[i]+f_Mg[i]+f_Na[i]+f_K[i]+f_H[i]))
                       
            #initial guess
            Alk0 = (Alk_tot[i]-R_alk[i-1])/(n*Zr*s[i]*1000)
            CO2_w0 = IC_tot[i]/(n*Zr*1000)*1/(s[i]*(1+k1[i]/H[i-1]+k2[i]*k1[i]/(H[i-1]**2))+(1-s[i])/k_H[i]) 
            R_alk0 = R_alk[i-1]
            Al_w0 = (Al_tot[i]-(f_Al[i-1]/3)*CEC_tot*conv_Al)/(n*Zr*s[i]*1000)#s[i-1]*Al_w[i-1]/s[i]
            Al0 = (H[i-1]**4/(H[i-1]**4+H[i-1]**3*K1+H[i-1]**2*K1*K2+H[i-1]*K1*K2*K3+K1*K2*K3*K4))*Al_w0
            Mg0 = (Mg_tot[i]-f_Mg[i-1]/2*CEC_tot)/(n*Zr*s[i]*1000) #s[i-1]*Mg[i-1]/s[i] 
            Na0 = (Na_tot[i]-f_Na[i-1]*CEC_tot)/(n*Zr*s[i]*1000) #s[i-1]*Na[i-1]/s[i] 
            Ca0 = (Ca_tot[i]-f_Ca[i-1]/2*CEC_tot)/(n*Zr*s[i]*1000) #s[i-1]*Ca[i-1]/s[i]
            K0 =  (K_tot[i]-f_K[i-1]*CEC_tot)/(n*Zr*s[i]*1000) #s[i-1]*K[i-1]/s[i]
            H0 = H[i-1]
            def eqH(p):
                    H0 = p
                    return((k1[i]*CO2_w0/H0+2*k1[i]*k2[i]*CO2_w0/(H0**2)-H0+k_w[i]/H0)-Alk0)
            H0_2 =fsolve(eqH, H[i-1])[0]
            
            #solution 1
            x0 = np.array([Alk0, CO2_w0, H0, R_alk0, Al_w0, Al0, Mg0, Ca0, Na0, K0, f_Al[i-1],f_Mg[i-1], f_Na[i-1], f_K[i-1], f_H[i-1], f_Ca[i-1]])         
            sol = fsolve(equations,x0, xtol=1e-8)                                           
            errors[:,i] = equations(sol) #residuals
            
            #solution 2 (robust reporting)
            res_threshold = 1e-1
            if np.any(abs(errors[:,i]) > res_threshold) or (not np.all(np.isfinite(errors[:,i]))):
                x0 = np.array([Alk0, CO2_w0, H0_2, R_alk0, Al_w0, Al0, Mg0, Ca0, Na0, K0, f_Al[i-1],f_Mg[i-1], f_Na[i-1], f_K[i-1], f_H[i-1], f_Ca[i-1]])
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    sol, info, ier, mesg = fsolve(equations, x0, xtol=1e-10, full_output=True)
                errors[:,i] = equations(sol)
                max_res = np.nanmax(np.abs(errors[:,i])) if np.all(np.isfinite(errors[:,i])) else np.nan
                if (ier != 1) or (not np.all(np.isfinite(sol))) or (not np.all(np.isfinite(errors[:,i]))) or (max_res > res_threshold):
                    # FALLBACK: restore previous timestep values instead of crashing
                    # fsolve's equation() writes to arrays as side-effect, so restore clean state
                    Alk[i] = Alk[i-1]; CO2_w[i] = CO2_w[i-1]; H[i] = H[i-1]
                    R_alk[i] = R_alk[i-1]; Al_w[i] = Al_w[i-1]; Al[i] = Al[i-1]
                    Mg[i] = Mg[i-1]; Ca[i] = Ca[i-1]; Na[i] = Na[i-1]; K[i] = K[i-1]
                    f_Al[i] = f_Al[i-1]; f_Mg[i] = f_Mg[i-1]; f_Na[i] = f_Na[i-1]
                    f_K[i] = f_K[i-1]; f_H[i] = f_H[i-1]; f_Ca[i] = f_Ca[i-1]
                    n_solver_failures += 1
                    if n_solver_failures > max_solver_failures:
                        raise ValueError(f"Solution not converging: {n_solver_failures} failures in {i} timesteps ({100*n_solver_failures/i:.1f}%)")

            #pH and C
            pH[i] = -np.log10(H[i]/conv_mol) # [-]
            CO2_air[i] = CO2_w[i]/k_H[i] #[mol/l]
            HCO3[i] = k1[i]*CO2_w[i]/H[i] 
            CO3[i] = k2[i]*k1[i]*CO2_w[i]/(H[i]**2)
            DIC[i] = CO2_w[i]+HCO3[i]+CO3[i]
            
            # Calculate concentrations needed for monitoring
            Si[i] = Si_tot[i]/(n*Zr*s[i]*1000) # [mol-conv/l]
            # An[i] already set above (constant concentration)
            
            # MONITORING: Print every 24 timesteps (1 day) if monitoring enabled
            if monitor_progress and i % 24 == 0 and i > 0:
                day = i // 24
                year = day / 365.25
                # Only print every 30 days to avoid spam
                if day % 30 == 0:
                    import sys
                    print(f"   {day:6d} {year:6.2f} {r_het[i]:10.1f} {pH[i]:6.2f} "
                          f"{Ca[i]:10.1f} {Mg[i]:10.1f} {Alk[i]:10.1f} {An[i]:10.1f} {s[i]:8.3f}", flush=True)
                    sys.stdout.flush()  # Force immediate output
              
            #Al speciation
            AlOH[i] = (H[i]**3*K1/(H[i]**4+H[i]**3*K1+H[i]**2*K1*K2+H[i]*K1*K2*K3+K1*K2*K3*K4))*Al_w[i] #[mol/l]
            AlOH2[i] = (H[i]**2*K1*K2/(H[i]**4+H[i]**3*K1+H[i]**2*K1*K2+H[i]*K1*K2*K3+K1*K2*K3*K4))*Al_w[i]
            AlOH3[i] = (H[i]*K1*K2*K3/(H[i]**4+H[i]**3*K1+H[i]**2*K1*K2+H[i]*K1*K2*K3+K1*K2*K3*K4))*Al_w[i]
            AlOH4[i] = Al_w[i]-(Al[i]+AlOH[i]+AlOH2[i]+AlOH3[i])
            
            # Note: Si[i] and An[i] are now calculated earlier (before monitoring) for display purposes
                                                 
            #CO2 diff flux 
            Fs[i] = D[i]/(Z_CO2)*(CO2_air[i]-CO2_atm)*1000 # [mol/d]    
            
            #Carbonate minerals
            CaCO3[i] = CaCO3[i-1] - W_CaCO3[i-1]*dt # [mol-conv]
            MgCO3[i] = MgCO3[i-1] - W_MgCO3[i-1]*dt

            if reset_carbonates_at_step is not None and i == reset_carbonates_at_step:
                CaCO3[i] = 0.0
                MgCO3[i] = 0.0

            #Carbonate weathering
            Omega_CaCO3[i] = Ca[i]*CO3[i]/K_CaCO3 # [-]
            Omega_MgCO3[i] = Mg[i]*CO3[i]/K_MgCO3
            [W_CaCO3[i], W_MgCO3[i]] = pyEW.carb_W(CaCO3[i], MgCO3[i], Omega_CaCO3[i], Omega_MgCO3[i], s[i], Zr, r_CaCO3,r_MgCO3,tau_CaCO3,tau_MgCO3)
                     
            #Silicate weathering
            if M_rock_in > 0:
                if i in tt_app:
                    # A questo timestep viene applicata nuova roccia.
                    prev_mass = M_rock[i-1]
                    M_rock[i] = prev_mass + M_rock_in
                    applied_cum += M_rock_in
                    for j in range(0, number_min):
                        M_min[j, i] = M_min[j, i-1] + rock_f_in[j] * M_rock_in
                    if M_rock[i] > 0:
                        rock_f[:, i] = M_min[:, i] / M_rock[i]
                    else:
                        rock_f[:, i] = 0 # Or np.zeros(number_min) depending on desired shape
                    print(f"[ROCK] t={i} applied={M_rock_in:.1f} prev_mass={prev_mass:.1f} new_mass={M_rock[i]:.1f} cumulative_applied={applied_cum:.1f}")

                    d[:, i] =  d_in   # oppure potresti voler combinare d[i-1] e la nuova roccia
                    delta_d[:, i] = np.insert(np.diff(d[:, i]), 0, d[0, i])
                    
                    # SAFETY CHECK: Prevent division by zero in psd calculation at application
                    psd[:, i] = np.where(delta_d[:, i] > 1e-12, 
                                        psd_perc_in * M_rock[i] / delta_d[:, i], 0)
                    
                    if SSA_in > 0:
                        # SAFETY CHECK: Prevent division by zero in fractal constant calculation
                        denominator = np.sum(d[:, i]**(b-1) * psd[:, i] * delta_d[:, i])
                        if denominator > 1e-12:
                            a = (SSA_in * rho_rock * M_rock[i] / 6) / denominator
                        else:
                            a = (1/(2*1e-10))**b  # Use default fractal constant
                    
                    lamb[:, i] = a * d[:, i]**b
                    # SAFETY CHECK: Prevent division by zero in SSA calculation at application
                    SSA[:, i] = np.where(d[:, i] > 1e-12, 
                                        6 / (d[:, i] * rho_rock) * lamb[:, i], 0)
                    SA[i] = np.sum(SSA[:, i] * psd[:, i] * delta_d[:, i])
                    for j in range(0, number_min):
                        Omega[j, i] = pyEW.Omega_sil(mineral[j], Ca[i], Mg[i], K[i], Na[i], Si[i], H[i], K_sp[j], conv_mol)
                        Wr[j, i] = s[i] * diss_f * (k_diss_H_t[j, i] * (H[i] / conv_mol)**n_H[j] +
                                                    k_diss_w_t[j, i] +
                                                    k_diss_OH_t[j, i] * (H[i] / conv_mol)**n_OH[j]) * (1 - Omega[j, i])
                        EW[j, i] = Wr[j, i] * SA[i] * rock_f[j, i]


                else:
                    # Aggiornamento weathering per i timesteps non di applicazione
                    
                    d_shrink = np.sum(rock_f[:, i-1] * Wr[:, i-1] * MM_min[:] / rho_rock) * dt
                    d[:, i] = d[:, i-1] - 2 * d_shrink * lamb[:, i-1]
                    d[:, i][d[:, i] < 0] = 0
                    delta_d[:, i] = np.insert(np.diff(d[:, i]), 0, d[0, i])
                    
                    for k in range(0, n_d_cl):
                        if d[k, i] > 0:
                            lamb[k, i] = a * d[k, i]**b
                            # SAFETY CHECK: Prevent division by zero in SSA calculation
                            if d[k, i] > 1e-12:  # Minimum particle diameter threshold
                                SSA[k, i] = 6 / (d[k, i] * rho_rock) * lamb[k, i]
                                # SAFETY CHECK: Prevent division by zero in psd calculation
                                if d[k, i-1] > 1e-12 and delta_d[k, i] > 1e-12:
                                    psd[k, i] = psd[k, i-1] * (d[k, i] / d[k, i-1])**3 * (delta_d[k, i-1] / delta_d[k, i])
                                else:
                                    psd[k, i] = 0  # Set to zero if division would cause NaN
                            else:
                                SSA[k, i] = 0
                                psd[k, i] = 0
                        else:
                            lamb[k, i] = 0
                            SSA[k, i] = 0
                            psd[k, i] = 0
                    
                    for j in range(0, number_min):
                        if M_min[j, i-1] != 0:
                            Omega[j, i] = pyEW.Omega_sil(mineral[j], Ca[i], Mg[i], K[i], Na[i], Si[i], H[i], K_sp[j], conv_mol)
                            Wr[j, i] = s[i] * diss_f * (k_diss_H_t[j, i] * (H[i] / conv_mol)**n_H[j] +
                                                        k_diss_w_t[j, i] +
                                                        k_diss_OH_t[j, i] * (H[i] / conv_mol)**n_OH[j]) * (1 - Omega[j, i])
                            M_min[j, i] = M_min[j, i-1] - EW[j, i-1] * MM_min[j] * dt
                            if M_min[j, i] < 0:
                                M_min[j, i] = 0
                    
                    SA[i] = np.sum(SSA[:, i] * psd[:, i] * delta_d[:, i])
                    
                    # SAFETY CHECK: Prevent NaN in M_rock calculation
                    M_rock_new = np.sum(M_min[:, i]) + (M_rock[i-1] - np.sum(M_min[:, i-1]))
                    if np.isnan(M_rock_new) or np.isinf(M_rock_new):
                        # Use fallback: just sum of current mineral masses
                        M_rock[i] = max(0, np.sum(M_min[:, i]))
                    else:
                        M_rock[i] = max(0, M_rock_new)  # Ensure non-negative
                    
                    # SAFETY CHECK: Prevent division by zero in rock_f calculation
                    if M_rock[i] > 1e-12:  # Minimum rock mass threshold
                        rock_f[:, i] = M_min[:, i] / M_rock[i]
                        # Additional safety: ensure rock fractions sum to reasonable value
                        total_fraction = np.sum(rock_f[:, i])
                        if total_fraction > 1.5 or total_fraction < 0.1:
                            # Reset to equal fractions if unreasonable
                            rock_f[:, i] = np.ones(number_min) / number_min * min(total_fraction, 1.0)
                    else:
                        rock_f[:, i] = 0  # Set all fractions to zero
                        
                    # SAFETY CHECK: Validate EW calculation
                    EW_new = Wr[:, i] * SA[i] * rock_f[:, i]
                    # Check for NaN or extreme values in EW
                    if np.any(np.isnan(EW_new)) or np.any(np.isinf(EW_new)):
                        EW[:, i] = 0  # Set to zero if invalid
                    else:
                        EW[:, i] = EW_new
                    

    data = {k: v for k, v in locals().items()}
                               
    return data

