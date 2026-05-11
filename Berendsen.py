from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.langevin import Langevin
from ase.units import fs, bar, kB
from mace.calculators import MACECalculator
import os
import glob
import sys
import traceback
import time
import numpy as np

def print_system_info(atoms, step_name):
    """Print system information for debugging"""
    try:
        energy = atoms.get_total_energy()
        forces = atoms.get_forces()
        max_force = max(abs(forces.flatten())) if len(forces) > 0 else 0.0
        volume = atoms.get_volume()
        cell = atoms.get_cell()
        temperature = atoms.get_temperature() if hasattr(atoms, 'get_temperature') else 0.0

        print(f"  📊 {step_name}:")
        print(f"    Energy: {energy:.3f} eV")
        print(f"    Max force: {max_force:.3f} eV/Å")
        print(f"    Volume: {volume:.2f} Å³")
        print(f"    Cell: [{cell[0,0]:.2f}, {cell[1,1]:.2f}, {cell[2,2]:.2f}] Å")
        print(f"    Temperature: {temperature:.1f} K")

        # Check for velocities if available
        if hasattr(atoms, 'get_velocities'):
            try:
                velocities = atoms.get_velocities()
                if velocities is not None:
                    max_vel = max(abs(velocities.flatten()))
                    print(f"    Max velocity: {max_vel:.3f} Å/fs")
            except:
                pass

    except Exception as e:
        print(f"  ⚠️ Could not get system info for {step_name}: {e}")

def calculate_energies(atoms):
    """Calculate total, potential, and kinetic energies"""
    try:
#        total_energy = atoms.get_total_energy()
        potential_energy = atoms.get_total_energy()  # Total energy from calculator is potential
        
        # Calculate kinetic energy from velocities
        kinetic_energy = 1.5 * 0.00008615 * atoms.get_temperature()
        total_energy = potential_energy + kinetic_energy 
#        if hasattr(atoms, 'get_velocities'):
#            velocities = atoms.get_velocities()
#            if velocities is not None:
#                masses = atoms.get_masses()
                # Convert masses from amu to eV·fs²/Å² units
                # 1 amu = 931.494 MeV/c² = 931.494e6 eV/c²
                # c = 2.998e8 m/s = 2.998e18 Å/s
                # 1 fs = 1e-15 s
                # Mass conversion factor: 931.494e6 eV / (2.998e18 Å/s)² * (1e-15 s/fs)²
#                mass_conversion = 931.494e6 / (2.998e18)**2 * (1e-15)**2
                
#                for i in range(len(atoms)):
#                    v = velocities[i]
#                    m = masses[i] * mass_conversion
#                    kinetic_energy += 0.5 * m * np.dot(v, v)
        
        return total_energy, potential_energy, kinetic_energy
    except Exception as e:
        print(f"Warning: Could not calculate energies: {e}")
        return 0.0, 0.0, 0.0

def write_log_header(log_file):
    """Write header for log file"""
    with open(log_file, 'w') as f:
        f.write("# Step\tTime(fs)\tTemperature(K)\tPressure(bar)\tVolume(Å³)\t")
        f.write("Total_Energy(eV)\tPotential_Energy(eV)\tKinetic_Energy(eV)\n")

def write_log_entry(log_file, step, time_fs, atoms, dynamics=None):
    """Write a single log entry"""
    try:
        temperature = atoms.get_temperature() if hasattr(atoms, 'get_temperature') else 0.0
        volume = atoms.get_volume()
        total_energy, potential_energy, kinetic_energy = calculate_energies(atoms)
        
        # Try to get pressure from dynamics object
        pressure = 0.0
        if dynamics is not None:
            try:
                if hasattr(dynamics, 'get_pressure'):
                    pressure = dynamics.get_pressure()   
                    print(f"pressure in dynamics")
                elif hasattr(dynamics, 'pressure_au'):
                    pressure = dynamics.pressure_au() 
                    print(f"pressure_au in dynamics")
            except:
                pass
        
        with open(log_file, 'a') as f:
            f.write(f"{step}\t{time_fs:.3f}\t{temperature:.2f}\t{pressure:.6f}\t{volume:.2f}\t")
            f.write(f"{total_energy:.6f}\t{potential_energy:.6f}\t{kinetic_energy:.6f}\n")
            print(f"{step} {time_fs:.3f} {temperature:.2f} {pressure:.6f} {volume:.2f} {total_energy:.6f} {potential_energy:.6f} {kinetic_energy:.6f}")
    except Exception as e:
        print(f"Warning: Could not write log entry: {e}")

def safe_md_run(dynamics, steps, output_path, log_file, step_name, save_interval, progress_interval):
    """Safely run MD with comprehensive error handling, progress tracking, and logging"""
    print(f"\n🚀 Starting {step_name} ({steps} steps)...")
    start_time = time.time()
    
    # Initialize log file
    write_log_header(log_file)
    
    try:
        # Clear trajectory file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Save initial frame and log entry
        write(output_path, dynamics.atoms, append=True)
        write_log_entry(log_file, 0, 0.0, dynamics.atoms, dynamics)
        
        # Get timestep for time calculation
        timestep = dynamics.dt / fs  # Convert to fs
        
        for step in range(steps):
            try:
                # Run single MD step
                dynamics.run(1)
                
                # Calculate current time
                current_time = (step + 1) * timestep
                
                # Save trajectory at specified intervals
                if (step + 1) % save_interval == 0:
                    write(output_path, dynamics.atoms, append=True)
                    write_log_entry(log_file, step + 1, current_time, dynamics.atoms, dynamics)
                
                # Print progress
                if (step + 1) % progress_interval == 0:
                    elapsed = time.time() - start_time
                    progress = (step + 1) / steps * 100
                    eta = elapsed / (step + 1) * (steps - step - 1)
                    
                    # Get current system info
                    temp = dynamics.atoms.get_temperature() if hasattr(dynamics.atoms, 'get_temperature') else 0.0
                    energy = dynamics.atoms.get_total_energy()
                    
                    print(f"  ⏳ Step {step+1}/{steps} ({progress:.1f}%) - T={temp:.1f}K - E={energy:.3f}eV")
                    print(f"      Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
                
            except Exception as e:
                print(f"❌ {step_name} failed at step {step+1}: {e}")
                print(f"Full traceback:")
                traceback.print_exc()
                
                # Save current state before failing
                error_file = output_path.replace('.traj', f'_error_step_{step+1}.xyz')
                error_log = log_file.replace('.log', f'_error_step_{step+1}.log')
                try:
                    write(error_file, dynamics.atoms)
                    write_log_entry(error_log, step + 1, (step + 1) * timestep, dynamics.atoms, dynamics)
                    print(f"  💾 Saved error state to: {error_file}")
                    print(f"  💾 Saved error log to: {error_log}")
                except:
                    print(f"  ⚠️ Could not save error state")
                
                raise e
        
        # Save final frame and log entry
        write(output_path, dynamics.atoms, append=True)
        write_log_entry(log_file, steps, steps * timestep, dynamics.atoms, dynamics)
        
        elapsed = time.time() - start_time
        print(f"✅ {step_name} completed successfully in {elapsed:.1f}s")
        
        # Print final trajectory info
        print(f"  📁 Trajectory saved to: {output_path}")
        print(f"  📊 Log file saved to: {log_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ {step_name} failed completely: {e}")
        return False

# ---------- Find .xyz File in Current Directory ----------
print("🔍 Looking for input file...")
xyz_files = glob.glob("*.xyz")
file_names = [os.path.splitext(f)[0] for f in xyz_files]
FF=file_names[0]

if not xyz_files:
    raise FileNotFoundError("❌ No .xyz file found in the current directory.")
elif len(xyz_files) > 1:
    raise RuntimeError(f"⚠️ Multiple .xyz files found: {xyz_files}. Please ensure only one is present.")

xyz_input = xyz_files[0]
filename = os.path.splitext(xyz_input)[0]
print(f"✅ Found input file: {xyz_input}")

# ---------- Configuration ----------
mace_model_path = "/home/arnab/scratch1/After13May/Rad_iso/structures_from_smiles/MACE_Model/mace.model"
base_output_dir = f"md_outputs_{filename}"

# Verify MACE model exists
if not os.path.exists(mace_model_path):
    raise FileNotFoundError(f"❌ MACE model not found at: {mace_model_path}")

print(f"📁 Creating output directory: {base_output_dir}")
os.makedirs(base_output_dir, exist_ok=True)

# Create subfolders
dirs = {
    'min': os.path.join(base_output_dir, "minimization"),
    'nve': os.path.join(base_output_dir, "nve"),
    'npt_eq': os.path.join(base_output_dir, "npt_equil"),
    'nvt_prod': os.path.join(base_output_dir, "nvt_prod")
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# ---------- Simulation Parameters ----------
print("⚙️ Setting simulation parameters...")

# MD simulation parameters
TEMPERATURE = 300  # K - Temperature for NVE, NPT, and NVT
FMAX = 0.5  # 0.5 eV/Å - Force convergence criterion for minimization
NVE_STEPS = 500  # Number of NVE equilibration steps = 750 fs
NPT_STEPS = 1000  # Number of NPT equilibration steps = 1500 fs
NVT_STEPS = 5000  # Number of NVT production steps = 7500 fs

print(f"  Temperature: {TEMPERATURE} K")
print(f"  Force convergence: {FMAX} eV/Å")
print(f"  NVE steps: {NVE_STEPS}")
print(f"  NPT steps: {NPT_STEPS}")
print(f"  NVT steps: {NVT_STEPS}")

# ---------- Load System and Setup Calculator ----------
print("📖 Loading system...")
try:
    atoms = read(xyz_input)
    atoms.set_pbc([True, True, True])
    atoms.set_cell([60, 60, 60])
    print(f"System loaded: {len(atoms)} atoms")
    print_system_info(atoms, "Initial system")
except Exception as e:
    print(f"❌ Failed to load system: {e}")
    sys.exit(1)

# Create calculator once and reuse throughout
print("🧮 Initializing MACE calculator...")
try:
    calc = MACECalculator(
        model_path=mace_model_path,
        # device="cuda",  # Uncomment if GPU available
        # default_dtype="float32"  # Uncomment for slight speedup
    )
    atoms.calc = calc

    # Test calculator
    initial_energy = atoms.get_total_energy()
    print(f"✅ Calculator working. Initial energy: {initial_energy:.3f} eV")
except Exception as e:
    print(f"❌ Calculator initialization failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------- Step 1: Structure Minimization ----------
print("\n🔽 Starting structure minimization...")
try:
    ucf = UnitCellFilter(atoms, constant_volume=True)
    dyn_min = BFGS(ucf, logfile=os.path.join(dirs['min'], "minimization.log"))
    dyn_min.run(fmax=FMAX)

    write(os.path.join(dirs['min'], "minimized.xyz"), atoms)
    print("✅ Minimization completed")
    print_system_info(atoms, "After minimization")

except Exception as e:
    print(f"❌ Minimization failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------- Step 2: Short NVE Equilibration ----------
print("\n🔥 Starting short NVE equilibration...")
try:
    # Set initial velocities (only once!)
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
    print(f"✅ Initial velocities set for {TEMPERATURE}K")

    dyn_nve = VelocityVerlet(atoms, timestep=1.5 * fs)
    nve_traj_path = os.path.join(dirs['nve'], "nve_equil.traj")
    nve_log_path = os.path.join(dirs['nve'], "nve_equil.log")

    success = safe_md_run(dyn_nve, NVE_STEPS, nve_traj_path, nve_log_path, "NVE equilibration",
                         1, 5)

    if success:
        write(os.path.join(dirs['nve'], "nve_final.xyz"), atoms)
        print_system_info(atoms, "After NVE")
    else:
        print("❌ NVE equilibration failed, stopping simulation")
        sys.exit(1)

except Exception as e:
    print(f"❌ NVE equilibration failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------- Pressure Setup for NPT ----------
print("\n🌡️ Setting up NPT pressure...")
pressure_au = 1.01325 
compressibility_au = 4.57e-5 

# ---------- Step 3: Short NPT Equilibration ----------
print("\n🌡️ Starting short NPT equilibration...")
try:
    # Don't reinitialize velocities - continue from NVE
    dyn_npt_eq = NPTBerendsen(
        atoms,
        timestep=1.5 * fs,
        temperature_K=TEMPERATURE,
        pressure_au=pressure_au,
        compressibility_au=compressibility_au,
        taut=100 * fs,
        taup=200 * fs,
    )

    npt_eq_traj_path = os.path.join(dirs['npt_eq'], "npt_equil.traj")
    npt_eq_log_path = os.path.join(dirs['npt_eq'], "npt_equil.log")

    success = safe_md_run(dyn_npt_eq, NPT_STEPS, npt_eq_traj_path, npt_eq_log_path, "NPT equilibration",
                         1, 5)

    if success:
        write(os.path.join(dirs['npt_eq'], "npt_equil_final.xyz"), atoms)
        print_system_info(atoms, "After NPT equilibration")
    else:
        print("❌ NPT equilibration failed, stopping simulation")
        sys.exit(1)

except Exception as e:
    print(f"❌ NPT equilibration failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------- Step 4: NVT Production Run ----------
print("\n🏃 Starting NVT production run...")
try:
    # Option A: NVT with Berendsen thermostat (fastest, good for most purposes)
    dyn_nvt_prod = NVTBerendsen(
        atoms,
        timestep=1.5 * fs,
        temperature_K=TEMPERATURE,
        taut=100 * fs,
    )

    # Option B: NVT with Langevin dynamics (better sampling, slightly slower)
    # Uncomment below and comment above if you prefer Langevin
    # dyn_nvt_prod = Langevin(
    #     atoms,
    #     timestep=1.5 * fs,
    #     temperature_K=TEMPERATURE,
    #     friction=0.01,
    # )

    nvt_prod_traj_path = os.path.join(dirs['nvt_prod'], "nvt_prod.traj")
    nvt_prod_log_path = os.path.join(dirs['nvt_prod'], f"{FF}.log")

    success = safe_md_run(dyn_nvt_prod, NVT_STEPS, nvt_prod_traj_path, nvt_prod_log_path, "NVT production",
                         1, 2)

    if success:
        write(os.path.join(dirs['nvt_prod'], f"{FF}.xyz"), atoms)
        print_system_info(atoms, "After NVT production")
    else:
        print("❌ NVT production failed")
        sys.exit(1)

except Exception as e:
    print(f"❌ NVT production failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------- Final Summary ----------
print(f"\n🎉 All MD steps completed successfully!")
print(f"📁 Results saved in: {base_output_dir}")
print(f"📊 Final summary:")
print(f"  - Minimized structure: {dirs['min']}/minimized.xyz")
print(f"  - NVE trajectory: {dirs['nve']}/nve_equil.traj")
print(f"  - NVE log: {dirs['nve']}/nve_equil.log")
print(f"  - NPT equilibration: {dirs['npt_eq']}/npt_equil.traj")
print(f"  - NPT log: {dirs['npt_eq']}/npt_equil.log")
print(f"  - NVT production: {dirs['nvt_prod']}/nvt_prod.traj")
print(f"  - NVT log: {dirs['nvt_prod']}/nvt_prod.log")
print(f"  - Final structure: {dirs['nvt_prod']}/nvt_prod_final.xyz")
print(f"  - Pressure used: 1 atm")
print(f"\n📈 Expected trajectory frames:")
print(f"  - NVE: ~{NVE_STEPS//10 + 2} frames (every 10 steps + initial/final)")
print(f"  - NPT equilibration: ~{NPT_STEPS//25 + 2} frames (every 25 steps + initial/final)")
print(f"  - NVT production: ~{NVT_STEPS//50 + 2} frames (every 50 steps + initial/final)")
print(f"\n📊 Log files contain:")
print(f"  - Step number and time (fs)")
print(f"  - Temperature (K)")
print(f"  - Pressure (GPa)")
print(f"  - Volume (Å³)")
print(f"  - Total, potential, and kinetic energies (eV)")
print(f"\n⚡ Computational savings:")
print(f"  - Reduced NVE: {NVE_STEPS} vs 1500 steps ({(1500-NVE_STEPS)/1500*100:.0f}%)")
print(f"  - Reduced NPT: {NPT_STEPS} vs 5000 steps ({(5000-NPT_STEPS)/5000*100:.0f}%)")
print(f"  - NVT production: ~30-40% faster than NPT")
print(f"  - Total speedup: ~40-50% compared to full NPT protocol")
print(f"\n🔬 Analysis recommendations:")
print(f"  - Use NVT production trajectory for energy/structural analysis")
print(f"  - Density equilibrated during NPT phase")
print(f"  - Temperature controlled during NVT production")
print(f"  - Recommended: Analyze last {NVT_STEPS//2} steps for statistics")
