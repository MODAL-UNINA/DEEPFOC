C
C PostScript.for - A  (sub)set  of CalComp HCBS-compatible plotting routines for
C output to PostScript devices (laser printers, phototypesetters, film  writers,
C etc.) and writing Encapsulated PostScript files (including Adobe Illustrator).
C
C
C Entry points:
C
C      Call PLOTS (Arguments ignored)
C
C         PLOTS  is the first plotting routine called and must be called exactly
C         once by the program
C
C         CALL PLOT (0.0,0.0,999) terminates all plotting
C
C      Call PLOT (x,y,ipen)
C
C         x     = X-coordinate
C         y     = Y-coordinate
C         ipen  = Function code
C
C                 +999 = End of run
C                   +3 = Move to (X,Y)
C                   +2 = Draw to (X,Y)
C                   -3 = Move to (X,Y), re-origin
C                   -2 = Draw to (X,Y), re-origin
C                 -999 = End of frame
C
C                 All other values ignored
C
C      Call NEWPEN (ipen)
C
C         ipen  = Line width (1 to 20 pixels on an Apple LaserWriter)
C
C      Call LSTYLE (istyle)
C
C         istyle= Line style (1=solid, 2..5=dashed)
C
C      Call SYMBOL (x,y,hgt,itext,angle,nc)
C
C         x     = X-coordinate (999.=continue from last X position)
C         y     = Y-coordinate (999.=continue from last Y position)
C         hgt   = Character height
C         itext = Hollerith character string (nc>0), ASCII character code
C                    (nc=0), or integer symbol number (nc<0)
C         angle = Character angle
C         nc    = Number of characters (nc>=0) or integer symbol number
C                    (nc=-1, pen up, nc<-1, pen down)
C
C
C Notes:
C
C      1.  The  name  of the PostScript device is specified by PARAMETER DEVICE.
C          The following devices names are recognized:
C
C             LASERWRITER     Apple LaserWriter (300 dpi)
C             LINOTRONIC      Linotronic phototypesetter (7200 dpi)
C             EPSF            Encapsulated PostScript (7200 dpi)
C             ILLUSTRATOR     Adobe Illustrator (7200 dpi)
C
C          Any other value for DEVICE is treated as a generic PostScript printer
C          with arbitrary resolution (supplied in PARAMETER ( DEVDPI = n ).)
C
C      2.  The file name used to contain the PostScript commands is specified by
C          PARAMETER DEVNAM.  The  following  file  names  are  defined  in  the
C          include files:
C
C             LASERWRITER     LaserWriter.ps
C             LINOTRONIC      Linotronic.ps
C             EPSF            PostScript.epsf
C             ILLUSTRATOR     Illustrator.epsf
C
C          If a file name is not legal on your system, alter the line
C
C             Parameter  ( DEVNAM = ... )
C
C          to define an acceptable file name.
C
C      3.  The PostScript operators "grestoreall" and "initgraphics" are used in
C          the Document Setup portion of the Document Body to restore  an  Apple
C          LaserWriter  to its initial state for beginning a new document.  This
C          violates the Document Structuring Conventions, which prohibit the use
C          of these operators in a conforming document.  However, if there is no
C          document processor or filter program which manages the device,  there
C          is no alternative method to reinitialize the device (using software).
C          To convert the document to a conforming document, which can  then  be
C          safely  incorporated into another document, remove or comment-out the
C          offending line in the PostScript output file.  (There is a comment in
C          the file immediately before the line with these same instructions.)
C
C      4.  The Encapsulated PostScript File Format does not permit more than one
C          page in a document.  Therefore, the PostScript output file is  closed
C          at  the  end  of  the  first page, whether or not the plotting job is
C          finished.  If the program attempts to plot  another  page,  an  error
C          message is printed and the program is stopped.
C
C      5.  If your Fortran compiler does not support the VAX form of the INCLUDE
C          statement, replace the lines
C
C             Include    'PostScript.inc'
C
C          with the lines
C
C             Character  DEVICE*(*), DEVNAM*(*)
C             Integer    DEVDPI
C             Parameter  ( DEVICE = 'LASERWRITER' )
C             Parameter  ( DEVNAM = 'LaserWriter.ps' )
C             Parameter  ( DEVDPI = 300 )
C             Logical    ISEPSF, ISAI
C             Parameter  ( ISEPSF = DEVICE .eq. 'EPSF' .or.
C            1                      DEVICE .eq. 'ILLUSTRATOR' )
C             Parameter  ( ISAI   = DEVICE .eq. 'ILLUSTRATOR' )
C
C          to select the output device (in this case, an Apple LaserWriter), the
C          name of the output file (in this  case,  'LaserWriter.ps'),  and  the
C          resolution of the device (used only for generic PostScript printers).
C
C      6.  The  OPEN  statment  for DEVNAM contains the non-standard VAX keyword
C          option  CARRIAGECONTROL='LIST'  so  that  the  file  containing   the
C          PostScript  commands  can  be typed or edited like a normal text file
C          (no line is longer than 80 characters) and when it is "PRINT"ed,  all
C          characters  are  sent  to  the PostScript printer, including those in
C          column 1 (which would otherwise be interpreted as as Fortran carriage
C          control  characters  and  would  NOT be sent).  (An alternative is to
C          write a space character at the front of every  record  in  SUBROUTINE
C          FLUSH.)
C
C      7.  Unit numbers 1 and 2 are used for plotting.  These are defined by the
C          Parameters ICHANL and JCHANL.  If they conflict with  units  on  your
C          system, they can be changed to any legal Fortran unit numbers.
C
C      8.  System-specific  SUBROUTINES  USERID  and TSTAMP are called to obtain
C          the user name of the caller and the date and time of the run for  the
C          header records in the document (%%...) and to label each page.
C
C      9.  SUBROUTINE SYMBOL  silently  truncates the passed character string to
C          255 characters.
C
C     10.  SUBROUTINE SYMBOL  declares  its  passed (Hollerith) character string
C          using the VAX INTEGER*4 statement and uses a  64A4  FORMAT  statement
C          specification  to  perform  Hollerith-to-character string conversion.
C          These may have to be modified for your Fortran compiler.
C
C     11.  SUBROUTINE AFMBB returns the bounding-box of a character string.   It
C          uses the Adobe Font Metric information supplied by  BLOCK DATA AFMBD,
C          which is generated automatically by the Parse_AFM_File program.
C
C     12.  SUBROUTINE  SYMBOL does not take ligatures and character-pair kerning
C          of a font into account when computing the bounding-box or the X and Y
C          coordinates  for  the  special  continuation call (x=999. or y=999.).
C          (However, continuation is correctly encoded in the PostScript  output
C          file,  except  for  Adobe  Illustrator,  which  has no "currentpoint"
C          operator to save the X and Y coordinates for later restoration.)
C
C     13.  The  PostScript  font names used on each page and in the entire docu-
C          ment are  recorded  in  separate  1024-character  lists.   SUBROUTINE
C          APPLST silently ignores any names that will not fit.
C
C
C References:
C
C      Adobe Systems, Inc. publications:
C         PostScript Language Reference Manual, Second Edition (includes
C            Document Structuring Conventions -- Version 3.0 and
C            Encapsulated PostScript File Format -- Version 3.0)
C         PostScript Language Tutorial and Cookbook
C         PostScript Language Program Design
C         PostScript Language Supplement for the Linotronic Imagesetter
C            Version 47.1
C         Adobe Illustrator Document Format -- Version 2.0
C      California Computer Products, Inc. publications:
C         Programming CalComp Electromechanical Plotters
C
C
C Author:
C
C      Lawrence M. Baker
C      U.S. Geological Survey
C      345 Middlefield Road  MS977
C      Menlo Park, CA  94025
C      (415)329-5608 or FTS 459-5608
C
C
C                                  Disclaimer
C
C Although  this program has been tested by the Geological Survey, United States
C Department of the Interior, no warranty, expressed or implied, is made by  the
C Geological  Survey,  as  to  the  accuracy  and functioning of the program and
C related program material, nor shall the fact of  distribution  constitute  any
C such  warranty,  and  no responsibility is assumed by the Geological Survey in
C connection therewith.
C
C
C Modifications:
C
C      V1.3  Insert "grestoreall" and "initgraphics" in preamble in case
C               previous job aborts w/o issuing "grestore."
C      V2.0  Consolidate original PostScript.for (for Apple LaserWriters) and
C               Linotronic.for (for Linotronic phototypesetters) into a generic
C               PostScript.for which uses PostScript.inc to specify the device.
C            Remove the lines added in V1.3.
C            Implement Document Structuring Conventions -- Version 2.1.
C            Implement Encapsulated PostScript File Format -- Version 2.0.
C            Implement relative moves, draws ("v"->"rmoveto", "r"->"rlineto").
C      V2.1  %%BoundingBox cannot specify "(atend)" in Encapsulated PostScript
C               Files -- must re-read PostScript file to update %%BoundingBox.
C            Change resolution of Encapsulated Postscript Files from 4096 dpi to
C               7200 dpi (100X PostScript default user space).
C            Change private abbreviations to those used by Adobe Illustrator
C               ("d"->"L", "s"->"S", "newpen"->"w").
C            Restore the lines added in V1.3 for documents intended to be sent
C               directly to the device, i.e., non-Encapsulated PostScript Files.
C      V2.2  Adobe Illustrator ignores commands inside a %%BeginSetup/%%EndSetup
C               construct.  %%BeginSetup/%%EndSetup were removed so it would
C               recognize commands to set rounded end cap and line joins.
C      V2.3  Always issue a "showpage" to undo the effect of the transformation
C               in "everypage", even if something has not been written to the
C               page.  (E.g., QPLOT writes an empty initial frame with only a
C               call to NEWPEN in it.)
C      V2.4  (Incomplete) support for passthru of character strings (SYMBOL
C               entry point).
C      V2.5  Correct Encapsulated PostScript to include "everypage" for proper
C               scaling, etc.
C      V3.0  Implement Document Structuring Conventions -- Version 3.0.
C            Implement Encapsulated PostScript File Format -- Version 3.0.
C            Complete implementation of passthru of character strings (SYMBOL
C               entry point).
C            Implement Adobe Font Metric File parser (Parse_AFM_File.for) to
C               automate the creation of BLOCK DATA AFMBD containing bounding-
C               box dimensions used by SUBROUTINE AFMBB.
C            Increase common use of Adobe Illustrator syntax.
C            Add DEVDPI to include file for generic PostScript printers.
C      V3.1  Replace common Hershey fonts with equivalent PostScript fonts.
C            Take pen width into account when computing bounding-box.
C            Use Adobe Illustrator syntax to select fonts and draw text.
C            Use %!PS-Adobe-2.0 EPSF-1.2 for Illustrator documents to conform to
C               currently published specification and to ease import to other
C               third-party applications.
C            Remove Implicit None to aid in portability.
C      V3.2  Improvements to SUBROUTINE SYMBOL:
C               Escape open and closing parends in a string to avoid confusion
C                  with premature closing of a PostScript string object.
C               Substitute PostScript "enspace" character for "hyphen" to be
C                  more consistent with the appearance of a Hershey "hyphen."
C               Escape non-printable characters in the PostScript text object
C               Special case space character to appear more like Hershey.
C               Add entry points CENTXT, RGTTXT for centered and right- justi-
C                  fied text strings, respectively, to SUBROUTINE SYMBOL.
C            Add ipen = -2, -3 to SUBROUTINE PLOT.
C      V3.3  Bracket drawing of character strings in SUBROUTINE SYMBOL with VM
C               "save" and "restore" to avoid using up virtual memory.
C      V3.4  More complete solution to VM exhaustion than V3.3.  V3.3 merely
C               delayed the exhaustion of VM because it only reclaimed the VM
C               used by character strings, but did not reclaim the VM used by
C               the transformation matrix.  To completely recover all the VM
C               used to draw character strings, the PostScript array constructor
C               operator ("]") was redefined to look ahead for the "start of
C               text block" operator ("e") and to "save" VM before creating the
C               transformation matrix.  The "end of text block" operator ("T")
C               was modified to "restore" VM, thus reclaiming the VM used by
C               both the transformation matrix and the character string in the
C               text block.
C            Enclose each page in a "save"/"restore" pairing.  ("everypage" does
C               the "save" in %%BeginPageSetup; the "restore" is explicitly done
C               just before "showpage".)
C            Comment out the crude error handler -- DEC's PathWorks PostScript
C               print symbiont does a much better job now.
C            Correctly include kerning of space character in calculation of
C               text string bounding-box in SUBROUTINE SYMBOL.
C            Let PostScript device calculate extra kerning for space character
C               (/spx) on the fly when font is selected.
C      V3.5  Fix ipen = -3 in SUBROUTINE PLOT for first call to new frame.
C            Save/restore X and Y origin offsets around expansion of markers in
C               SUBROUTINE SYMBOL to avoid adding them twice to each coordinate.
C            Move ICHANL/JCHANL to include file so they can be dynamically
C               allocated by FUNCTION ALLOCU (non-Viewer version only).
C      V3.6  Add SUBROUTINE LSTYLE, mainly for SAC graphics file translator
C               program, SACVIEW.
C
C
      Subroutine npPLOTS (ibuf, nloc, ldev)
C
C     Include    'PostScript.inc'
C
C PostScript.com -- Include file for PostScript device selection at run-time.
C
      Character  DEVICE*32, DEVNAM*80
      Integer    DEVDPI, ICHANL, JCHANL
      Logical    ISEPSF, ISAI
      Common /PSDEV1/ DEVICE, DEVNAM
      Common /PSDEV2/ DEVDPI, ICHANL, JCHANL, ISEPSF, ISAI
      Save   /PSDEV1/, /PSDEV2/
C
C
      Character  VMAJOR, VMINOR, VERSN*3
      Parameter  ( VMAJOR  = '3' )
      Parameter  ( VMINOR  = '6' )
      Parameter  ( VERSN   = VMAJOR // '.' // VMINOR )
C
      Integer    ibuf(*), nloc, ldev
C
      Integer    ipctr
      Logical    isopen
      Common /IOC0M/  ipctr, isopen
      Save   /IOC0M/
C
C...  If DRAW or NEWPEN, virgin: .TRUE. -> .FALSE. (new page)
C...  If DRAW, dirty: .FALSE. -> .TRUE.
      Logical    island, virgin, dirty, ismove
      Integer    dpi, page, kount, pfonch, dfonch
      Real       pxlow, pylow, pxhi, pyhi, dxlow, dylow, dxhi, dyhi
      Real       x999, y999, bbpen, pscale, spx, orgx, orgy
      Common /PSC0M/  island, virgin, dirty, ismove, dpi, page, kount,
     1                pxlow, pylow, pxhi, pyhi, dxlow, dylow, dxhi,
     2                dyhi, x999, y999, bbpen, pfonch, dfonch, pscale,
     3                spx, orgx, orgy
      Character  pfonts*1024, dfonts*1024
      Common /PSC0N/  pfonts, dfonts
      Save   /PSC0M/, /PSC0N/
C
      Integer    j, ntitle, ncreat, nfor
      Logical    isused
      Character  title*80, create*32, for*80
      Save       isused
      Data       isused/.FALSE./
C
C
C...  Ignore all but the first call
C
      If (isused) Then
         Goto 9000
      End If
C
C...  Setup PostScript output device parameters in PostScript.com
C
      Call PSENV
C
C...  Open plot file for PostScript plotting characters
C
      Open (Unit=ICHANL,File=DEVNAM,Status='Unknown',Form='Formatted')
C
C...  Open temporary plot file for Encapsulated PostScript Files
C...  (cannot use "%%BoundingBox: (atend)")
C
      If (ISEPSF) Then
         Open (Unit=JCHANL,Status='Scratch',Form='Unformatted')
      End If
C
C...  Reset buffer pointer
      ipctr = 1
C
C
C...  Use Document Structuring Conventions Specification Version 3.0  ..........
C
C
C...  Start of Document Header  ................................................
C
C
      If (ISAI) Then
         Call PUTLIN ('%!PS-Adobe-2.0 EPSF-1.2',23)
      Else If (ISEPSF) Then
         Call PUTLIN ('%!PS-Adobe-3.0 EPSF-3.0',23)
      Else
         Call PUTLIN ('%!PS-Adobe-3.0',14)
      End If
      Call PUTLIN ('%%Creator: USGS Viewer.PostScript V'//VERSN,38)
      Call PUTSTR ('%%For:',6)
c      Call USERID (for)
      for = 'unknown user'
      Do 1100 j = LEN(for),1,-1
         If (for(j:j) .ne. ' ') Then
            Goto 1200
         End If
 1100    Continue
 1200 nfor = j
      Call PUTLIN (for,nfor)
      Call PUTSTR ('%%Title:',8)
      Inquire (Unit=ICHANL,Name=title)
      Do 1300 j = LEN(title),1,-1
         If (title(j:j) .ne. ' ') Then
            Goto 1400
         End If
 1300    Continue
 1400 ntitle = j
      Call PUTLIN (title,ntitle)
      Call PUTSTR ('%%CreationDate:',15)
      Call TSTAMP (create)
      Do 1500 j = LEN(create),1,-1
         If (create(j:j) .ne. ' ') Then
            Goto 1600
         End If
 1500    Continue
 1600 ncreat = j
      Call PUTLIN (create,ncreat)
      If (.not. ISAI) Then
         Call PUTLIN ('%%DocumentData: Clean7Bit',25)
      End If
      Call PUTLIN ('%%DocumentNeededResources: (atend)',34)
      If (.not. ISAI) Then
         Call PUTLIN ('%%DocumentSuppliedResources: procset ' //
     1                'USGS_Viewerdict ' // VMAJOR // ' ' // VMINOR,56)
      End If
      If (.not. ISEPSF) Then
         Call PUTLIN ('%%Orientation: Landscape',24)
      End If
C...  This line is replaced by the correct limits for Encapsulated PostScript
C...  Files (including Adobe Illustrator) when the plot is done
      Call PUTLIN ('%%BoundingBox: (atend)',22)
      If (ISAI) Then
         Call PUTLIN ('%%TemplateBox:0 0 0 0',21)
      End If
      If (.not. ISAI) Then
         If (ISEPSF) Then
            Call PUTLIN ('%%Pages: 0',10)
         Else
            Call PUTLIN ('%%Pages: (atend)',16)
            Call PUTLIN ('%%PageOrder: Ascend',19)
         End If
      End If
      Call PUTLIN ('%%EndComments',13)
C
C
C...  End of Document Header  ..................................................
C
C
C...  Start of Document Prologue  ..............................................
C
C
C...  Device-specific prologue
C
C...  Assume portrait (or unspecified) orientation
      island = .FALSE.
C
      If (ISAI) Then
C
C...     Adobe Illustrator (Encapsulated PostScript subset)
C
         dpi = 7200
C
      Else
C
         Call PUTLIN ('%%BeginProlog',13)
C
C...     Create a private dictionary called "USGS_Viewerdict"
C
         Call PUTLIN ('%%BeginResource: procset USGS_Viewerdict ' //
     1                VMAJOR // ' ' // VMINOR,44)
         Call PUTLIN ('userdict /USGS_Viewerdict 32 dict dup begin put',
     1                47)
C
         If (ISEPSF) Then
C
C...        Encapsulated PostScript
C
            dpi = 7200
C
         Else
C
            Call PUTLIN ('% Conditionally define "setstrokeadjust" ' //
     1                   'for Level 1 devices',60)
            Call PUTLIN ('/setstrokeadjust where',22)
            Call PUTLIN ('   { pop }',10)
            Call PUTLIN ('   { /setstrokeadjust /pop load def }',37)
            Call PUTLIN ('   ifelse',9)
C
            If (DEVICE .eq. 'LINOTRONIC') Then
C
C...           Linotronic phototypesetter (1270 dpi, or better)
C
               dpi = 7200
C
C...           Set page size to 11 by 17 inches, 1/2 inch offset (on 12 inch
C...           wide paper), landscape orientation, with origin in the lower left
C...           corner, positive X from left to right, positive Y from bottom to
C...           top.
C
C...           Linotronic private verb in statusdict:
C
C...              width height offset orientation - setpageparams -
C
C                 \------------------------------------------------\
C                 /     -      +------------------------------+    /
C                 \     ^      |                              |    \
C                 /   width    |  <--       height       -->  |    /
C                 \     v     ^|                              |^   \
C                 /     -     y+------------------------------+x   /
C                 \     ^      x-> =0     orientation    =1 <-y    \
C                 /   offset                                       /
C                 \     v               paper motion ->            \
C                 /------------------------------------------------/
C
               Call PUTLIN ('/device_setup { statusdict begin',32)
               Call PUTLIN ('% Change "false" to "true" in the' //
     1                      ' following line to produce' //
     2                      ' negative images',75)
               Call PUTLIN ('   /negativeprint false def ' //
     1                      '/mirrorprint false def',50)
               Call PUTLIN ('   11 72 mul 17 72 mul .5 72 mul ' //
     1                      '0 setpageparams',48)
               Call PUTLIN ('   end } bind def',17)
C
C...           Setup transformation for portrait orientation, assuming a 17 X 11
C...           page size
               Call PUTLIN ('/portrait {',11)
               Call PUTLIN ('   0 11 72 mul translate -90 rotate',35)
               Call PUTLIN ('   } bind def',13)
C...           No transformation for landscape orientation
               Call PUTLIN ('/landscape { } def',18)
C...           Force landscape orientation (for now)
               island = .TRUE.
C
            Else
C
C...           Apple LaserWriter (300 dpi) and generic PostScript printers
C
               dpi = DEVDPI
C
C...           By default, the page is in portrait orientation, with origin in
C...           the lower left corner, positive X from left to right, positive
C...           Y from bottom to top.
C
               Call PUTLIN ('/device_setup {',15)
               Call PUTLIN ('% Remove the next line to make ' //
     1                      'this a conforming document',57)
               Call PUTLIN ('% (i.e., to incorporate this document ' //
     1                      'into another document)',60)
               Call PUTLIN ('   grestoreall initgraphics',27)
               Call PUTLIN ('   } bind def',13)
C
*              Call PUTLIN ('/ident (' // for(1:nfor) // '  ' //
*    1                      create(1:ncreat) // ') def',15+nfor+ncreat)
*              Call PUTLIN ('/print_ident {',14)
*              Call PUTLIN ('   /Courier findfont 6 scalefont setfont',
*    1                      40)
*              Call PUTLIN ('   .75 72 mul 10.75 72 mul moveto ' //
*    1                      'ident show',44)
*              Call PUTLIN ('   } bind def',13)
C
C...           No transformation for portrait orientation
               Call PUTLIN ('/portrait { } def',17)
C...           Setup transformation for landscape orientation, assuming an
C...           8.5 X 11 page size
               Call PUTLIN ('/landscape {',12)
               Call PUTLIN ('   8.5 72 mul 0 translate 90 rotate',35)
               Call PUTLIN ('   } bind def',13)
C...           Force landscape orientation (for now)
               island = .TRUE.
C
            End If
C
         End If
C
C...     Start of commands issued at the start of every page
C
         Call PUTLIN ('/x999 0 def',11)
         Call PUTLIN ('/y999 0 def',11)
C...     Scale transformation matrix from 72/in to dpi (user=device coords)
         Call PUTLIN ('/everypage {',12)
C...     Save start-of-page VM state
         If (.not. ISEPSF) Then
            Call PUTLIN ('   save',7)
         End If
         Call PUTSTR ('   72',5)
         Call PUTINT (dpi)
         Call PUTLIN ('div dup scale',13)
         If (.not. ISEPSF) Then
C...        Automatic stroke adjustment (Level 2 devices only)
            Call PUTLIN ('   true setstrokeadjust',23)
         End If
C...     Use rounded end caps and inflection elbows for polylines
         Call PUTLIN ('   1 setlinecap 1 setlinejoin',29)
         Call PUTLIN ('   /x999 0 def /y999 0 def',26)
         Call PUTLIN ('   } bind def',13)
C
C...     End of commands issued at the start of every page
C
C...     Define abbreviations for "moveto", "rmoveto", "lineto", "rlineto",
C...     "stroke", "newpen", "setfont", and "setdash" using Adobe Illustrator's
C...     mnemonics
C
         Call PUTLIN ('% Basic plotting operators',26)
         Call PUTLIN ('/d /setdash load def',20)
         Call PUTLIN ('/L /lineto load def',19)
         Call PUTLIN ('/m /moveto load def',19)
         Call PUTLIN ('/r /rlineto load def',20)
         Call PUTLIN ('/S /stroke load def',19)
         Call PUTLIN ('/v /rmoveto load def',20)
         Call PUTLIN ('/w /setlinewidth load def',25)
C
C...     Define abbreviations for calls to SYMBOL
C
         Call PUTLIN ('% Text plotting operators',25)
         Call PUTLIN ('/spx 0 def',10)
         Call PUTLIN ('/xkern 0 def',12)
C...     Redefine the array construction operator "]" to look-ahead for
C...     the "start of text block" operator "e" and "save" VM is found.
C...     This is tricky, so here's a walkthrough of the stack:
C...        (]) cvn cvx {
C...           currentfile token { % mark any_1 ... any_n any_token
C...              dup              % mark any_1 ... any_n any_token any_token
C...              counttomark      % mark any_1 ... any_n any_token any_token n+2
C...              1 add 1 roll     % any_token mark any_1 ... any_n any_token
C...              /e eq {          % /e mark any_1 ... any_n
C...                 save          % /e mark any_1 ... any_n save
C...                 counttomark   % /e mark any_1 ... any_n save n+1
C...                 2 add 1 roll  % save /e mark any_1 ... any_n
C...                 } if
C...              ]                % any_token array  or  save /e array
C...              exch             % array any_token  or  save array /e
C...              exec             %                      save
C...              } if
C...           } bind def
         Call PUTLIN ('(]) cvn cvx {',13)
         Call PUTLIN ('   currentfile token {',22)
         Call PUTLIN ('      dup counttomark 1 add 1 roll',34)
         Call PUTLIN ('      /e eq { save counttomark 2 add 1 roll ' //
     1                '} if',48)
         Call PUTLIN ('      ] exch exec',17)
         Call PUTLIN ('      } if',10)
         Call PUTLIN ('   } bind def',13)
         Call PUTLIN ('/e {',4)
         Call PUTLIN ('   matrix currentmatrix exch concat ' //
     1                '0 0 moveto',46)
         Call PUTLIN ('   } bind def',13)
         Call PUTLIN ('/t {',4)
         Call PUTLIN ('   spx 0 32 xkern 0 6 -1 roll awidthshow pop',44)
         Call PUTLIN ('   } bind def',13)
         Call PUTLIN ('/T {',4)
         Call PUTLIN ('   setmatrix currentpoint 3 -1 roll restore ' //
     1                '/y999 exch def /x999 exch def',73)
         Call PUTLIN ('   } bind def',13)
         Call PUTLIN ('/z {',4)
         Call PUTLIN ('   pop /xkern exch def pop exch findfont ' //
     1                'exch scalefont setfont',63)
         Call PUTLIN ('   /spx (0) stringwidth pop ( ) stringwidth ' //
     1                'pop sub def',55)
         Call PUTLIN ('   } bind def',13)
C
         Call PUTLIN ('end',3)
         Call PUTLIN ('%%EndResource',13)
C
C...     End of definitions in private dictionary "USGS_Viewerdict"
C
C
      End If
C
C
C...  End of Document Prologue  ................................................
C
C
      Call PUTLIN ('%%EndProlog',11)
C
C
C...  Start of Document Body  ..................................................
C
C
C...  Start of Document Setup  .................................................
C
C
      If (.not. ISAI) Then
         Call PUTLIN ('%%BeginSetup',12)
         If (ISEPSF) Then
            Call PUTLIN ('%%IncludeResource: font',23)
*        Else
*           Call PUTLIN ('% Error handler',15)
*           Call PUTLIN ('/str 128 string def',19)
*           Call PUTLIN ('errordict begin',15)
*           Call PUTLIN ('   /handleerror {',17)
*           Call PUTLIN ('      $error /newerror get {',28)
*           Call PUTLIN ('         $error /newerror false put',35)
*           Call PUTLIN ('         showpage',17)
*           Call PUTLIN ('         /Courier findfont 12 scalefont ' //
*    1                   'setfont',57)
*           Call PUTLIN ('         0 0 moveto (errorname: ) show ' //
*    1                   '$error /errorname get str cvs show',73)
*           Call PUTLIN ('         0 -14 moveto (command  : ) show ' //
*    1                   '$error /command get str cvs show',73)
*           Call PUTLIN ('         showpage',17)
*           Call PUTLIN ('      } if',10)
*           Call PUTLIN ('   } bind def',13)
*           Call PUTLIN ('end',3)
         End If
         Call PUTLIN ('USGS_Viewerdict begin',21)
         If (.not. ISEPSF) Then
            Call PUTLIN ('device_setup',12)
         End If
         Call PUTLIN ('%%EndSetup',10)
      End If
C
C
C...  End of Document Setup  ...................................................
C
C
C...  Initialize variables used by PLOT entry points
      page   = 0
      virgin = .TRUE.
C...  Prevent another call to PLOTS
      isused = .TRUE.
C...  Allow file to be written by PLOT entry points now
      isopen = .TRUE.
C
 9000 Return
C
      End


