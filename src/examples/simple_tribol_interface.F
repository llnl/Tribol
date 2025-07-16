! Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
! other Tribol Project Developers. See the top-level LICENSE file for details.
!
! SPDX-License-Identifier: (MIT)
!-------------------------------------------------------------------------------
! Example demonstrating usage of the tribol_simple interface in Fortran
!-------------------------------------------------------------------------------

module simple_tribol_interface_ex

  use iso_c_binding, only : C_DOUBLE
  use mpi
  use tribol_simple
  use tribol_test_mesh

  implicit none

! TODO: Fix this -- tribol needs to determine if it should use mpif_header
#if defined AXOM_USE_MPI && defined AXOM_USE_MPIF_HEADER
  include 'mpif.h'
#endif

  integer :: rank, nprocs, ierr
  logical :: should_init_slic = .TRUE.

  ! Mesh data
  type(testmesh) mesh
  real(C_DOUBLE), allocatable :: mortar_gaps(:), mortar_pressures(:)

contains

  subroutine init_mpi()
    !...initialize MPI
    call mpi_init( ierr )
    call mpi_comm_rank( MPI_COMM_WORLD, rank, ierr )
    call mpi_comm_size( MPI_COMM_WORLD, nprocs, ierr )
  end subroutine init_mpi

  subroutine finalize_mpi()
    !... finalize MPI
    call mpi_finalize( ierr )
  end subroutine finalize_mpi

  !-------------------------------------------------------------------------------
  ! initialize_tribol(  )
  !
  ! Initializes tribol and internal logger (slic)
  ! Sets up a hex mesh and uses the simple coupling scheme
  !-------------------------------------------------------------------------------
  subroutine initialize_tribol()
    implicit none

    integer :: rc

    ! Mesh parameters
    integer :: N = 5
    integer :: dim, contact_method
  
    integer :: nElem1_x, nElem1_y, nElem1_z
    real(C_DOUBLE) :: min1_x, min1_y, min1_z
    real(C_DOUBLE) :: max1_x, max1_y, max1_z

    integer :: nElem2_x, nElem2_y, nElem2_z
    real(C_DOUBLE) :: min2_x, min2_y, min2_z
    real(C_DOUBLE) :: max2_x, max2_y, max2_z

    real(C_DOUBLE) :: theta1 = 0., theta2 = 0.

    real(C_DOUBLE), pointer :: pos_x(:), pos_y(:), pos_z(:)
    integer, pointer :: mortar_face_conn(:), nonmortar_face_conn(:)

    real(C_DOUBLE) :: area_frac = 1.e-03

    ! set up the test mesh
    ! TODO: Export the contact_method enum type
    dim = 3
    contact_method = 2 ! tribol::MORTAR_WEIGHTS

    nElem1_x = N
    nElem1_y = N
    nElem1_z = N
    
    nElem2_x = N
    nElem2_y = N
    nElem2_z = N

    min1_x = 0. 
    min1_y = 0.
    min1_z = 0. 

    max1_x = 1.
    max1_y = 1. 
    max1_z = 1.05 

    min2_x = 0. 
    min2_y = 0. 
    min2_z = .95 

    max2_x = 1. 
    max2_y = 1. 
    max2_z = 2. 

    !! Initialize tribol with  some basic error checking
    if (rank == 0) then
      write(*,*) "** Initializing tribol"
    end if

    rc = tribol_simple_initialize(dim, should_init_slic)

    if ( rc /= 0 ) then 
      error stop "tribol initialization failed!"
    endif


    !! Create the mesh object
    if (rank == 0) then
      write(*,*) "** Initializing hex test mesh"
    end if

    mesh = tribol_test_mesh_new()

    call mesh%setup_contact_mesh_hex( &
      nElem1_x, nElem1_y, nElem1_z, &
      min1_x, min1_y, min1_z, &
      max1_x, max1_y, max1_z, &
      nElem2_x, nElem2_y, nElem2_z, &
      min2_x, min2_y, min2_z, &
      max2_x, max2_y, max2_z, &
      theta1, theta2 )

    ! Output some info about the generated mesh
    if (rank == 0) then
      write(*,*) "Mortar mesh:"
      write(*,*) "- Num faces", mesh%get_nummortarfaces()
      write(*,*) "- Num nodes", mesh%get_numtotalnodes()

      write(*,*) "Nonmortar mesh:"
      write(*,*) "- Num faces", mesh%get_numnonmortarfaces()
      write(*,*) "- Num nodes", mesh%get_numtotalnodes()
    end if


    !! Use the generated mesh to set up a simple couple scheme
    if (rank == 0) then
      write(*,*) "** Setting up tribol simple coupling"
    end if
  
    ! Initialize arrays
    pos_x => mesh%get_x()
    pos_y => mesh%get_y()
    pos_z => mesh%get_z()
    mortar_face_conn => mesh%get_mortar_face_connectivity()
    nonmortar_face_conn  => mesh%get_nonmortar_face_connectivity()

    ! Allocate and zero out necessary fields for mortar computation
    allocate( mortar_gaps( mesh%get_numtotalnodes() ) )
    mortar_gaps = mortar_gaps*0.0

    allocate( mortar_pressures( mesh%get_numtotalnodes() ) )
    mortar_pressures = mortar_pressures*0.0


    ! Set up simple coupling scheme
    call tribol_simple_setup_coupling( &
        dim, contact_method, &
        mesh%get_nummortarfaces(), mesh%get_numtotalnodes(), &
        mortar_face_conn, pos_x, pos_y, pos_z, &
        mesh%get_numnonmortarfaces(), mesh%get_numtotalnodes(), &
        nonmortar_face_conn, pos_x, pos_y, pos_z, &
        area_frac, mortar_gaps, mortar_pressures )

  end subroutine initialize_tribol

  subroutine finalize_tribol()
    implicit none

    integer :: rc

    deallocate( mortar_gaps )
    deallocate( mortar_pressures )

    if (rank == 0) then
      write(*,*) "** Finalizing tribol"
    end if

    rc = tribol_simple_finalize(should_init_slic)

  end subroutine finalize_tribol


  subroutine get_mortar_weights()
    implicit none

    integer :: rc
    real(C_DOUBLE) :: dt = 1.0
 
    ! CSR data
    integer(C_INT), pointer :: csr_I(:), csr_J(:)
    real(C_DOUBLE), pointer :: csr_vals(:)

    if (rank == 0) then
      write(*,*) "** Calling tribol update"
    end if

    ! Run tribol on the static configuration
    rc = tribol_simple_update(dt)

    !...apply some basic error checking
    if ( rc /= 0 ) then 
      error stop "tribol update failed!"
    endif


    if (rank == 0) then
      write(*,*) "** Getting mortar weights"
    end if

    rc = tribol_simple_get_coupling_CSR(csr_I,csr_J,csr_vals)


    if (rank == 0) then
      write(*,*) "Num offsets in CSR  (size of I) ", size(csr_I)
      write(*,*) "Num nonzeros in CSR (size of J) ", size(csr_J)
      write(*,*) "                 (size of vals) ", size(csr_vals)
    end if


  end subroutine get_mortar_weights

end module simple_tribol_interface_ex


!-------------------------------------------------------------------------------
! Main Program
!-------------------------------------------------------------------------------
program tribol_simple_interface_fortran_ex

  use simple_tribol_interface_ex
  implicit none

  call init_mpi()

  call initialize_tribol()
  call get_mortar_weights()

  call finalize_tribol()
  call finalize_mpi()

end program
