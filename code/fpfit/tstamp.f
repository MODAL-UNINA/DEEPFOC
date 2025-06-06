      Subroutine TSTAMP ( string )
C
C...  This is standard Fortran-90
C
      Implicit   None
C
C...  Format is MM/DD/YYYY hh:mm:ss.sss
      Character  string*(*)
C
      Character  date*8, time*10
C
C
C...  date is CCYYMMDD, time is hhmmss.sss
      Call Date_and_Time( date, time )
      string = date(5:6) // '/' // date(7:8) // '/' // date(1:4) //
     1         ' ' // time(1:2) // ':' // time(3:4) // ':' // time(5:10)
C
      Return
C
      End
