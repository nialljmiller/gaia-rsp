! Automated Gaia RSP fitting runner.
!
! x_integer_ctrl(1) selects the stage:
!   1 = LINA screen, then settle a nonlinear RSP model
!   2 = run Colors for x_integer_ctrl(5) additional completed periods
!   3 = LINA-only evaluation; always terminate at model 0 after writing LINA_RESULT
!
! Stage 1 rejects the requested mode when its LINA growth is not greater than
! x_ctrl(3). A settled model must also retain a nonzero, stable rsp_DeltaR or
! rsp_DeltaMag; decay to zero amplitude is not accepted as a limit cycle.

module run_star_extras

   use star_lib
   use star_def
   use const_def
   use math_lib
   use utils_lib, only: is_bad
   use auto_diff

   implicit none

   include "test_suite_extras_def.inc"

   integer, parameter :: mode_settle = 1
   integer, parameter :: mode_colors = 2
   integer, parameter :: mode_lina_only = 3

   integer, parameter :: max_lina_modes = 16

   integer :: last_checked_period = -1
   integer :: stable_period_count = 0
   integer :: colors_start_period = -1
   logical :: lina_checked = .false.
   logical :: have_previous_amplitude = .false.
   real(dp) :: previous_delta_r = -1d0
   real(dp) :: previous_delta_mag = -1d0

   ! LINA periods retained from the pre-screen so that the settled nonlinear
   ! period can be checked against the mode that was actually requested.
   ! Nonlinear mode selection is not guaranteed to follow the linear screen:
   ! when more than one mode is unstable the model can settle into a different
   ! one depending on the initial kick, and without this check a wrong-period
   ! limit cycle would be accepted and passed to Colors.
   integer :: saved_target_mode = -1
   integer :: saved_nmodes = 0
   real(dp) :: saved_lina_periods_days(max_lina_modes) = -1d0

contains

   include "test_suite_extras.inc"

   subroutine extras_controls(id, ierr)
      integer, intent(in) :: id
      integer, intent(out) :: ierr
      type(star_info), pointer :: s

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      s% extras_startup => extras_startup
      s% extras_check_model => extras_check_model
      s% extras_start_step => extras_start_step
      s% extras_finish_step => extras_finish_step
      s% extras_after_evolve => extras_after_evolve
      s% how_many_extra_history_columns => how_many_extra_history_columns
      s% data_for_extra_history_columns => data_for_extra_history_columns
      s% how_many_extra_profile_columns => how_many_extra_profile_columns
      s% data_for_extra_profile_columns => data_for_extra_profile_columns
   end subroutine extras_controls

   subroutine extras_startup(id, restart, ierr)
      integer, intent(in) :: id
      logical, intent(in) :: restart
      integer, intent(out) :: ierr
      type(star_info), pointer :: s

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      call test_suite_startup(s, restart, ierr)
      if (ierr /= 0) return

      last_checked_period = -1
      stable_period_count = 0
      colors_start_period = -1
      lina_checked = .false.
      have_previous_amplitude = .false.
      previous_delta_r = -1d0
      previous_delta_mag = -1d0
      saved_target_mode = -1
      saved_nmodes = 0
      saved_lina_periods_days(:) = -1d0
   end subroutine extras_startup

   integer function extras_start_step(id)
      integer, intent(in) :: id
      integer :: ierr, i, target_mode, target_index, run_mode
      real(dp) :: minimum_growth, selected_growth, selected_period_days
      type(star_info), pointer :: s

      extras_start_step = keep_going
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      run_mode = s% x_integer_ctrl(1)
      if (run_mode /= mode_settle .and. run_mode /= mode_lina_only) return
      if (lina_checked) return
      lina_checked = .true.

      target_mode = s% x_integer_ctrl(6)
      target_index = target_mode + 1
      minimum_growth = s% x_ctrl(3)

      write(*,'(A)')
      write(*,'(A)') 'LINA pre-screen for requested pulsation mode'
      write(*,'(A,I0)') 'available LINA modes: ', s% RSP_nmodes
      do i = 1, s% RSP_nmodes
         write(*,'(A,I0,A,1PE13.5,A,1PE13.5)') '  mode ', i - 1, &
            ': period_days=', s% rsp_LINA_periods(i)/secday, &
            ' growth=', s% rsp_LINA_growth_rates(i)
      end do

      if (target_mode < 0 .or. target_index > s% RSP_nmodes) then
         write(*,'(A,I0)') 'REJECT: requested LINA mode is unavailable: ', target_mode
         call write_lina_result(s, target_mode, minimum_growth, 'unavailable', &
            'requested mode is unavailable')
         if (run_mode == mode_settle) then
            call write_lina_rejection(s, target_mode, 0d0, minimum_growth, &
               'requested mode is unavailable')
         end if
         extras_start_step = terminate
         return
      end if

      selected_growth = s% rsp_LINA_growth_rates(target_index)
      selected_period_days = s% rsp_LINA_periods(target_index)/secday
      write(*,'(A,I0,A,1PE13.5,A,1PE13.5)') 'selected mode ', target_mode, &
         ': period_days=', selected_period_days, ' growth=', selected_growth
      write(*,'(A,1PE13.5)') 'minimum accepted growth: ', minimum_growth

      if (is_bad(selected_growth)) then
         call write_lina_result(s, target_mode, minimum_growth, 'invalid', &
            'selected growth is non-finite')
         if (run_mode == mode_settle) then
            call write_lina_rejection(s, target_mode, selected_growth, &
               minimum_growth, 'selected growth is non-finite')
         end if
         extras_start_step = terminate
         return
      end if

      if (selected_growth <= minimum_growth) then
         call write_lina_result(s, target_mode, minimum_growth, 'damped', &
            'selected mode is not linearly excited')
         if (run_mode == mode_lina_only) then
            write(*,'(A)') 'LINA-only result: selected mode is damped; terminate at model 0.'
         else
            write(*,'(A)') 'REJECT: selected mode is not linearly excited.'
            call write_lina_rejection(s, target_mode, selected_growth, &
               minimum_growth, 'selected mode is not linearly excited')
         end if
         extras_start_step = terminate
         return
      end if

      call write_lina_result(s, target_mode, minimum_growth, 'excited', &
         'selected mode is linearly excited')

      saved_target_mode = target_mode
      saved_nmodes = min(s% RSP_nmodes, max_lina_modes)
      do i = 1, saved_nmodes
         saved_lina_periods_days(i) = s% rsp_LINA_periods(i)/secday
      end do

      if (run_mode == mode_lina_only) then
         write(*,'(A)') 'LINA-only result: selected mode is linearly excited; terminate at model 0.'
         extras_start_step = terminate
         return
      end if

      write(*,'(A)') 'PASS: selected mode is linearly excited; begin nonlinear settling.'
      write(*,'(A)')
   end function extras_start_step

   integer function extras_finish_step(id)
      integer, intent(in) :: id
      integer :: ierr
      integer :: mode, min_periods, required_stable_periods
      integer :: max_settle_periods, target_color_periods
      integer :: nperiod, rel_period
      real(dp) :: grekm_tol, grekm_abs
      real(dp) :: min_delta_r, min_delta_mag, amplitude_rel_tol
      real(dp) :: delta_r, delta_mag, delta_r_rel, delta_mag_rel
      logical :: grekm_ok, radius_ok, magnitude_ok
      logical :: radius_stable, magnitude_stable, amplitude_stable
      integer :: nearest_mode
      real(dp) :: nonlinear_period_days, energy_tol, rel_run_E_err
      type(star_info), pointer :: s

      extras_finish_step = keep_going
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return

      mode = s% x_integer_ctrl(1)
      nperiod = s% rsp_num_periods

      select case (mode)

      case (mode_settle)
         min_periods = max(0, s% x_integer_ctrl(2))
         required_stable_periods = max(1, s% x_integer_ctrl(3))
         max_settle_periods = max(min_periods, s% x_integer_ctrl(4))

         grekm_tol = s% x_ctrl(2)
         if (grekm_tol <= 0d0) grekm_tol = 1d-3
         min_delta_r = max(0d0, s% x_ctrl(4))
         min_delta_mag = max(0d0, s% x_ctrl(5))
         amplitude_rel_tol = s% x_ctrl(6)
         if (amplitude_rel_tol <= 0d0) amplitude_rel_tol = 2d-2

         if (nperiod <= last_checked_period) return
         last_checked_period = nperiod

         grekm_abs = abs(s% rsp_GREKM)
         delta_r = abs(s% rsp_DeltaR)
         delta_mag = abs(s% rsp_DeltaMag)

         delta_r_rel = huge(1d0)
         delta_mag_rel = huge(1d0)
         radius_stable = .false.
         magnitude_stable = .false.

         radius_ok = min_delta_r > 0d0 .and. delta_r >= min_delta_r
         magnitude_ok = min_delta_mag > 0d0 .and. delta_mag >= min_delta_mag

         if (have_previous_amplitude) then
            if (radius_ok .and. previous_delta_r >= min_delta_r) then
               delta_r_rel = abs(delta_r - previous_delta_r) / &
                  max(delta_r, previous_delta_r, min_delta_r)
               radius_stable = delta_r_rel <= amplitude_rel_tol
            end if
            if (magnitude_ok .and. previous_delta_mag >= min_delta_mag) then
               delta_mag_rel = abs(delta_mag - previous_delta_mag) / &
                  max(delta_mag, previous_delta_mag, min_delta_mag)
               magnitude_stable = delta_mag_rel <= amplitude_rel_tol
            end if
         end if

         amplitude_stable = radius_stable .or. magnitude_stable
         grekm_ok = .not. is_bad(grekm_abs) .and. grekm_abs <= grekm_tol

         if (nperiod < min_periods) then
            if (mod(nperiod, 10) == 0) then
               write(*,'(A,I0,A,I0,A,1PE11.3,A,1PE11.3)') &
                  'settling RSP: period ', nperiod, ' of minimum ', min_periods, &
                  '; DeltaR=', delta_r, '; DeltaMag=', delta_mag
            end if
            previous_delta_r = delta_r
            previous_delta_mag = delta_mag
            have_previous_amplitude = .true.
            return
         end if

         if (grekm_ok .and. amplitude_stable) then
            stable_period_count = stable_period_count + 1
         else
            stable_period_count = 0
         end if

         write(*,'(A,I0,A,1PE11.3,A,L1,A,1PE11.3,A,1PE11.3)') &
            'settle check: period ', nperiod, ' abs(GREKM)=', grekm_abs, &
            ' ok=', grekm_ok, ' DeltaR=', delta_r, ' DeltaMag=', delta_mag
         write(*,'(A,1PE11.3,A,L1,A,1PE11.3,A,L1,A,I0,A,I0)') &
            '  rel_dR=', delta_r_rel, ' stable=', radius_stable, &
            ' rel_dMag=', delta_mag_rel, ' stable=', magnitude_stable, &
            ' accepted=', stable_period_count, '/', required_stable_periods

         previous_delta_r = delta_r
         previous_delta_mag = delta_mag
         have_previous_amplitude = .true.

         if (stable_period_count >= required_stable_periods) then
            call report_energy_error(s)

            ! The limit cycle is stable, but is it the mode we asked for? Compare
            ! the nonlinear period against the retained LINA periods and reject if
            ! another mode is the closer match.
            nonlinear_period_days = s% rsp_period/secday
            nearest_mode = nearest_lina_mode(nonlinear_period_days)

            if (nearest_mode >= 0 .and. saved_target_mode >= 0 .and. &
                nearest_mode /= saved_target_mode) then
               write(*,'(A)') 'REJECT: settled limit cycle is not the requested mode.'
               write(*,'(A,1PE13.5,A,I0,A,I0)') &
                  '  nonlinear period_days=', nonlinear_period_days, &
                  ' matches mode ', nearest_mode, ', requested mode ', saved_target_mode
               call write_mode_rejection(s, nonlinear_period_days, nearest_mode)
               extras_finish_step = terminate
               return
            end if

            ! Energy conservation is an acceptance criterion, not just a warning.
            energy_tol = s% x_ctrl(7)
            if (energy_tol <= 0d0) energy_tol = 1d-4
            rel_run_E_err = 0d0
            if (s% total_energy /= 0d0) then
               rel_run_E_err = s% cumulative_energy_error/s% total_energy
            end if

            if (is_bad(rel_run_E_err) .or. abs(rel_run_E_err) > energy_tol) then
               write(*,'(A,1PE13.5,A,1PE13.5)') &
                  'REJECT: energy error too large: ', rel_run_E_err, &
                  ' exceeds tolerance ', energy_tol
               call write_energy_rejection(s, rel_run_E_err, energy_tol)
               extras_finish_step = terminate
               return
            end if

            call write_settled_acceptance(s, grekm_abs, delta_r, delta_mag)
            write(*,'(A)') 'PASS: nonlinear nonzero-amplitude limit cycle settled.'
            write(*,'(A,1PE13.5,A,I0)') '  nonlinear period_days=', &
               nonlinear_period_days, ' consistent with mode ', saved_target_mode
            extras_finish_step = terminate
            return
         end if

         if (max_settle_periods > 0 .and. nperiod >= max_settle_periods) then
            call report_energy_error(s)
            call write_settling_rejection(s, grekm_abs, delta_r, delta_mag)
            write(*,'(A,I0,A)') 'REJECT: maximum settling period ', &
               max_settle_periods, ' reached without an accepted limit cycle.'
            extras_finish_step = terminate
            return
         end if

      case (mode_colors)
         target_color_periods = max(1, s% x_integer_ctrl(5))

         if (colors_start_period < 0) then
            colors_start_period = nperiod
            write(*,'(A,I0,A,I0,A)') 'Colors stage starts at completed period ', &
               colors_start_period, '; target is +', target_color_periods, ' periods.'
            return
         end if

         rel_period = nperiod - colors_start_period
         if (nperiod > last_checked_period) then
            last_checked_period = nperiod
            if (mod(rel_period, 5) == 0) then
               write(*,'(A,I0,A,I0)') 'Colors production: completed additional periods ', &
                  rel_period, '/', target_color_periods
            end if
         end if

         if (rel_period >= target_color_periods) then
            call report_energy_error(s)
            write(*,'(A,I0,A)') 'Completed Colors run of ', target_color_periods, &
               ' additional periods.'
            extras_finish_step = terminate
            return
         end if

      case (mode_lina_only)
         return

      case default
         return
      end select
   end function extras_finish_step

   ! Zero-based index of the LINA mode whose linear period is closest (in log)
   ! to the supplied nonlinear period. Returns -1 when no LINA data was retained.
   integer function nearest_lina_mode(period_days)
      real(dp), intent(in) :: period_days
      integer :: i
      real(dp) :: separation, best_separation

      nearest_lina_mode = -1
      if (saved_nmodes <= 0) return
      if (period_days <= 0d0 .or. is_bad(period_days)) return

      best_separation = huge(1d0)
      do i = 1, saved_nmodes
         if (saved_lina_periods_days(i) <= 0d0) cycle
         separation = abs(log10(period_days/saved_lina_periods_days(i)))
         if (separation < best_separation) then
            best_separation = separation
            nearest_lina_mode = i - 1
         end if
      end do
   end function nearest_lina_mode

   subroutine write_mode_rejection(s, period_days, nearest_mode)
      type(star_info), pointer :: s
      real(dp), intent(in) :: period_days
      integer, intent(in) :: nearest_mode
      integer :: io, i

      open(newunit=io, file='MODE_REJECTED', status='replace', action='write')
      write(io,'(A)') 'settled limit cycle is not the requested mode'
      write(io,'(A,I0)') 'period=', s% rsp_num_periods
      write(io,'(A,I0)') 'target_mode=', saved_target_mode
      write(io,'(A,I0)') 'settled_mode=', nearest_mode
      write(io,'(A,1PE17.9)') 'nonlinear_period_days=', period_days
      do i = 1, saved_nmodes
         write(io,'(A,I0,A,1PE17.9)') 'lina_mode_', i - 1, '_period_days=', &
            saved_lina_periods_days(i)
      end do
      close(io)
   end subroutine write_mode_rejection

   subroutine write_energy_rejection(s, rel_run_E_err, energy_tol)
      type(star_info), pointer :: s
      real(dp), intent(in) :: rel_run_E_err, energy_tol
      integer :: io

      open(newunit=io, file='ENERGY_REJECTED', status='replace', action='write')
      write(io,'(A)') 'cumulative energy error exceeds the accepted tolerance'
      write(io,'(A,I0)') 'period=', s% rsp_num_periods
      write(io,'(A,1PE17.9)') 'rel_run_E_err=', rel_run_E_err
      write(io,'(A,1PE17.9)') 'energy_tolerance=', energy_tol
      close(io)
   end subroutine write_energy_rejection

   subroutine write_lina_result(s, target_mode, minimum_growth, status, reason)
      type(star_info), pointer :: s
      integer, intent(in) :: target_mode
      real(dp), intent(in) :: minimum_growth
      character(len=*), intent(in) :: status, reason
      integer :: io, i, target_index
      real(dp) :: selected_growth, selected_period_days

      selected_growth = 0d0
      selected_period_days = 0d0
      target_index = target_mode + 1
      if (target_mode >= 0 .and. target_index <= s% RSP_nmodes) then
         selected_growth = s% rsp_LINA_growth_rates(target_index)
         selected_period_days = s% rsp_LINA_periods(target_index)/secday
      end if

      open(newunit=io, file='LINA_RESULT', status='replace', action='write')
      write(io,'(A)') 'status=' // trim(status)
      write(io,'(A)') 'reason=' // trim(reason)
      write(io,'(A,I0)') 'target_mode=', target_mode
      write(io,'(A,I0)') 'nmodes=', s% RSP_nmodes
      write(io,'(A,1PE17.9)') 'selected_period_days=', selected_period_days
      write(io,'(A,1PE17.9)') 'selected_growth=', selected_growth
      write(io,'(A,1PE17.9)') 'minimum_growth=', minimum_growth
      write(io,'(A,1PE17.9)') 'mass_msun=', s% RSP_mass
      write(io,'(A,1PE17.9)') 'luminosity_lsun=', s% RSP_L
      write(io,'(A,1PE17.9)') 'teff_k=', s% RSP_Teff
      write(io,'(A,1PE17.9)') 'x=', s% RSP_X
      write(io,'(A,1PE17.9)') 'z=', s% RSP_Z
      do i = 1, s% RSP_nmodes
         write(io,'(A,I0,A,1PE17.9)') 'mode_', i - 1, '_period_days=', &
            s% rsp_LINA_periods(i)/secday
         write(io,'(A,I0,A,1PE17.9)') 'mode_', i - 1, '_growth=', &
            s% rsp_LINA_growth_rates(i)
      end do
      close(io)
   end subroutine write_lina_result

   subroutine write_lina_rejection(s, target_mode, growth, minimum_growth, reason)
      type(star_info), pointer :: s
      integer, intent(in) :: target_mode
      real(dp), intent(in) :: growth, minimum_growth
      character(len=*), intent(in) :: reason
      integer :: io

      open(newunit=io, file='LINA_REJECTED', status='replace', action='write')
      write(io,'(A)') trim(reason)
      write(io,'(A,I0)') 'target_mode=', target_mode
      write(io,'(A,1PE17.9)') 'selected_growth=', growth
      write(io,'(A,1PE17.9)') 'minimum_growth=', minimum_growth
      write(io,'(A,1PE17.9)') 'mass_msun=', s% RSP_mass
      write(io,'(A,1PE17.9)') 'luminosity_lsun=', s% RSP_L
      write(io,'(A,1PE17.9)') 'teff_k=', s% RSP_Teff
      write(io,'(A,1PE17.9)') 'z=', s% RSP_Z
      close(io)
   end subroutine write_lina_rejection

   subroutine write_settling_rejection(s, grekm_abs, delta_r, delta_mag)
      type(star_info), pointer :: s
      real(dp), intent(in) :: grekm_abs, delta_r, delta_mag
      integer :: io

      open(newunit=io, file='SETTLING_REJECTED', status='replace', action='write')
      write(io,'(A)') 'maximum periods reached without an accepted limit cycle'
      write(io,'(A,I0)') 'period=', s% rsp_num_periods
      write(io,'(A,1PE17.9)') 'abs_rsp_GREKM=', grekm_abs
      write(io,'(A,1PE17.9)') 'rsp_DeltaR=', delta_r
      write(io,'(A,1PE17.9)') 'rsp_DeltaMag=', delta_mag
      close(io)
   end subroutine write_settling_rejection

   subroutine write_settled_acceptance(s, grekm_abs, delta_r, delta_mag)
      type(star_info), pointer :: s
      real(dp), intent(in) :: grekm_abs, delta_r, delta_mag
      integer :: io

      open(newunit=io, file='SETTLED_ACCEPTED', status='replace', action='write')
      write(io,'(A)') 'nonzero-amplitude limit cycle accepted'
      write(io,'(A,I0)') 'period=', s% rsp_num_periods
      write(io,'(A,1PE17.9)') 'abs_rsp_GREKM=', grekm_abs
      write(io,'(A,1PE17.9)') 'rsp_DeltaR=', delta_r
      write(io,'(A,1PE17.9)') 'rsp_DeltaMag=', delta_mag
      close(io)
   end subroutine write_settled_acceptance

   subroutine report_energy_error(s)
      type(star_info), pointer :: s
      real(dp) :: rel_run_E_err

      if (s% total_energy == 0d0) return
      rel_run_E_err = s% cumulative_energy_error/s% total_energy
      write(*,*) 'rel_run_E_err', rel_run_E_err
      if (abs(rel_run_E_err) > 1d-5) then
         write(*,*) '*** WARNING: large rel_run_E_error ***', rel_run_E_err
      end if
   end subroutine report_energy_error

   subroutine extras_after_evolve(id, ierr)
      integer, intent(in) :: id
      integer, intent(out) :: ierr
      type(star_info), pointer :: s

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      call test_suite_after_evolve(s, ierr)
   end subroutine extras_after_evolve

   integer function extras_check_model(id)
      integer, intent(in) :: id
      integer :: ierr
      type(star_info), pointer :: s

      extras_check_model = keep_going
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
   end function extras_check_model

   integer function how_many_extra_history_columns(id)
      integer, intent(in) :: id
      integer :: ierr
      type(star_info), pointer :: s

      how_many_extra_history_columns = 0
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
   end function how_many_extra_history_columns

   subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
      integer, intent(in) :: id, n
      character(len=maxlen_history_column_name) :: names(n)
      real(dp) :: vals(n)
      integer, intent(out) :: ierr
      type(star_info), pointer :: s

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
   end subroutine data_for_extra_history_columns

   integer function how_many_extra_profile_columns(id)
      use star_def, only: star_info
      integer, intent(in) :: id
      integer :: ierr
      type(star_info), pointer :: s

      how_many_extra_profile_columns = 0
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
   end function how_many_extra_profile_columns

   subroutine data_for_extra_profile_columns(id, n, nz, names, vals, ierr)
      use star_def, only: star_info, maxlen_profile_column_name
      use const_def, only: dp
      integer, intent(in) :: id, n, nz
      character(len=maxlen_profile_column_name) :: names(n)
      real(dp) :: vals(nz,n)
      integer, intent(out) :: ierr
      type(star_info), pointer :: s

      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
   end subroutine data_for_extra_profile_columns

end module run_star_extras
