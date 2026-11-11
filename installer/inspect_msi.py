"""Print the parts of BeamLab's .msi that decide what gets installed (read-only).
usage: python installer/inspect_msi.py dist/BeamLab-0.1.0.msi"""
import msilib
import sys

db = msilib.OpenDatabase(sys.argv[1], msilib.MSIDBOPEN_READONLY)


def rows(sql):
    v = db.OpenView(sql)
    v.Execute(None)
    out = []
    while True:
        r = v.Fetch()
        if r is None:
            break
        out.append([r.GetString(i) for i in range(1, r.GetFieldCount() + 1)])
    v.Close()
    return out


print('Features (Feature, Title, Level, Attributes):')
for r in rows('SELECT Feature, Title, Level, Attributes FROM Feature'):
    print('  ', r)
print('Feature conditions (Feature, Level, Condition):')
for r in rows('SELECT Feature_, Level, Condition FROM Condition'):
    print('  ', r)
print('Properties:')
for r in rows('SELECT Property, Value FROM Property'):
    if r[0] in ('INSTALLDRIVERS', 'ALLUSERS', 'ProductVersion', 'ProductName', 'UpgradeCode', 'ARPPRODUCTICON'):
        print('  ', r)
print('Custom action:')
for r in rows("SELECT Action, Type, Source, Target FROM CustomAction WHERE Action='RunDriverInstall'"):
    print('  ', r)
print('Sequenced as:')
for r in rows("SELECT Action, Condition, Sequence FROM InstallExecuteSequence WHERE Action='RunDriverInstall' OR Action='InstallFinalize'"):
    print('  ', r)
print('Shortcuts:')
for r in rows('SELECT Name, Directory_, Target, Arguments FROM Shortcut'):
    print('  ', r)
files = rows('SELECT FileName FROM File')
print('Files: %d, driver files: %s' % (len(files), [f[0].split('|')[-1] for f in files if 'driver' in f[0].lower() or f[0].lower().endswith('.msi')]))
print('Feature of the driver components:')
for r in rows('SELECT Feature_, Component_ FROM FeatureComponents'):
    if r[0] == 'Drivers':
        print('  ', r)
