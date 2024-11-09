import os
import psutil
import subprocess
import time
import ctypes
import sys
from ctypes import wintypes

# Constants
THREAD_SUSPEND_RESUME = 2
THREAD_QUERY_INFORMATION = 64
THREAD_ALL_ACCESS = 2032639
CONTEXT_FULL = 65543

# WinAPI setup
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.QueryThreadCycleTime.restype = wintypes.BOOL
kernel32.QueryThreadCycleTime.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_uint64)]
kernel32.SuspendThread.restype = wintypes.DWORD
kernel32.SuspendThread.argtypes = [wintypes.HANDLE]

# Thread functions
def open_thread(thread_id):
    return kernel32.OpenThread(THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION, False, thread_id)

def suspend_thread(thread_id):
    handle = open_thread(thread_id)
    if handle:
        try:
            result = kernel32.SuspendThread(handle)
            if result == (-1):
                raise Exception('Error suspending thread')
        finally:
            kernel32.CloseHandle(handle)

def resume_thread(thread_id):
    handle = open_thread(thread_id)
    if handle:
        try:
            result = kernel32.ResumeThread(handle)
            if result == (-1):
                raise Exception('Error resuming thread')
        finally:
            kernel32.CloseHandle(handle)

def close_handle(handle):
    kernel32.CloseHandle(handle)

# Process and thread management
def get_process_id_by_name(process_name):
    for proc in psutil.process_iter(['name', 'pid']):
        if proc.info['name'] == process_name:
            return proc.info['pid']
    return None

def get_thread_cycle_time(thread_id):
    handle = open_thread(thread_id)
    if handle:
        try:
            cycle_time = ctypes.c_uint64()
            result = kernel32.QueryThreadCycleTime(handle, ctypes.byref(cycle_time))
            if not result:
                raise Exception('Error querying thread cycle time')
            return cycle_time.value
        finally:
            close_handle(handle)
    return None

def list_all_thread_ids(process_name):
    process_id = get_process_id_by_name(process_name)
    if process_id:
        process = psutil.Process(process_id)
        return [thread.id for thread in process.threads()]
    return []

def list_thread_cycles(process_name):
    process_id = get_process_id_by_name(process_name)
    if process_id:
        process = psutil.Process(process_id)
        threads = process.threads()
        thread_cycles = {}

        for thread in threads:
            cycle_time = get_thread_cycle_time(thread.id)
            thread_cycles[thread.id] = cycle_time

        time.sleep(1)
        for thread in threads:
            current_cycle_time = get_thread_cycle_time(thread.id)
            delta = current_cycle_time - thread_cycles[thread.id]
            if delta == 0:
                suspend_thread(thread.id)

def suspend_threads_by_index(process_name):
    suspended_threads = []
    process_id = get_process_id_by_name(process_name)
    if process_id:
        process = psutil.Process(process_id)
        threads = process.threads()
        thread_cycles = {thread.id: get_thread_cycle_time(thread.id) for thread in threads}

        time.sleep(1)
        deltas = {}
        for thread in threads:
            current_cycle_time = get_thread_cycle_time(thread.id)
            delta = current_cycle_time - thread_cycles[thread.id]
            deltas[thread.id] = delta

        highest_delta_threads = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:2]
        for thread_id, _ in highest_delta_threads:
            suspend_thread(thread_id)
            suspended_threads.append(thread_id)
    return suspended_threads

# Service management
def start_service_and_wait(service_name):
    try:
        subprocess.check_call(['sc', 'start', service_name])
        print('Service started successfully.')
        return True
    except subprocess.CalledProcessError as e:
        print(f'Error starting service: {e}')
        return False

def stop_service(service_name):
    try:
        subprocess.check_call(['sc', 'stop', service_name])
        print('Service stopped successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Error stopping service: {e}')

# Utility functions
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def is_user_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False

def run_as_admin():
    script = sys.argv[0]
    params = ' '.join(sys.argv[1:])
    ctypes.windll.shell32.ShellExecuteW(None, 'runas', sys.executable, f'\"{script}\" {params}', None, 1)

def check_debugger():
    """Check for debugger presence."""
    if ctypes.windll.kernel32.IsDebuggerPresent() != 0:
        clear_screen()
        print("Debugger detected! Exiting.")
        sys.exit(1)

# Main program
def main():
    check_debugger()

    if not is_user_admin():
        print('Restarting as administrator...')
        run_as_admin()
        return

    process_name = 'vgc.exe'
    service_name = 'vgc'

    print('Checking if service is running...')
    time.sleep(3)
    clear_screen()

    if get_process_id_by_name(process_name):
        print(f'Service {service_name} is already running.')
        stop_service(service_name)
        print('Service stopped. Restarting...')
        time.sleep(5)
        clear_screen()

    service_started = start_service_and_wait(service_name)
    if service_started:
        thread_ids = list_all_thread_ids(process_name)
        time.sleep(3)
        recorded_threads = thread_ids[:3]  # Assuming target indices [5, 6, 8] were representative

        #while not get_process_id_by_name('VALORANT-Win64-Shipping.exe'):
        while not get_process_id_by_name('LeagueClient.exe'):
            print('Waiting for Valorant to start...')
            time.sleep(5)

        clear_screen()
        print('Valorant Running...')
        time.sleep(1)

        while True:
            clear_screen()
            print('Choose an option:')
            print('1. Run First')
            print('2. Run Second')
            print('3. Exit')

            choice = input('Enter your choice (1/2/3): ').strip()
            if choice == '1':
                clear_screen()
                print('Running popup bypass...')
                list_thread_cycles(process_name)
                print('Popup bypass completed.')
            elif choice == '2':
                clear_screen()
                print('Starting vgc bypass...')
                for thread_id in recorded_threads:
                    suspend_thread(thread_id)
                print('Bypass success')
            elif choice == '3':
                print('Exiting...')
                break
            else:
                print('Invalid choice. Please enter 1, 2, or 3.')

if __name__ == '__main__':
    main()
