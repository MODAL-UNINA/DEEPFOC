      subroutine readfl (inst, unit, filnm, status, erflag, prompt)

c reads file name from terminal and opens file

      character*80      inst                            
c							! (input) parameters on instruction line
      integer           unit                            
c							! (input) logical unit for reporting merge action
      character*40      filnm                           
c							! (input/output) name of file
      character*(*)     prompt                          
c							! (input) prompt string
      character*(*)     status                          
c							! (input) open status ('new' or 'old')
      integer           erflag                          
c							! (output) error flag; non-zero indicates unable to open

      integer           ios                             
c							! iostat specifier

      erflag = 0
      if (inst(1:10) .eq. '          ') then
        call askc (prompt, filnm)
      else
        read (inst, *, iostat = ios) filnm
        if (ios .ne. 0) then
          erflag = 1
          filnm = 'none'
        end if
      end if
      if (filnm(1:4) .ne. 'none') then
        if (status .eq. 'new') then
          open (unit, file = filnm, status = 'unknown', iostat = erflag)
c         open (unit, file = filnm, status = 'new', iostat = erflag,		! VAX/VMS version
c    & carriagecontrol = 'list', recl = 139)					! VAX/VMS version
        else
	  open (unit,file=filnm,status='old',blank='zero',iostat=erflag)
c         open (unit, file = filnm, status = 'old', blank = 'zero',		! VAX/VMS version
c    & iostat = erflag, readonly)						! VAX/VMS version
        end if
        if (erflag .ne. 0) filnm = 'none'
      end if
      if (filnm .eq. 'none') close (unit)
      return
      end
